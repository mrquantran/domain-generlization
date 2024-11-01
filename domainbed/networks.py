# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm
import torchvision.models
from torchvision import transforms
from transformers import ViTModel

import time
import numpy as np
import matplotlib.pyplot as plt

from domainbed.lib import wide_resnet
import copy

# CYCLEGAN Experiments
from domainbed.cyclegan.utils import get_sources


class LatentInterpolator(nn.Module):
    def __init__(
        self,
        num_domains,
        latent_dim,
        num_interpolation_points=5,
        consistency_lambda=0.1,
        diversity_lambda=0.05,
        cycle_lambda=0.1,
        momentum_beta=0.9,
        temperature=0.1,
    ):
        super(LatentInterpolator, self).__init__()
        self.num_domains = num_domains
        self.latent_dim = latent_dim
        self.num_points = num_interpolation_points
        self.consistency_lambda = consistency_lambda
        self.cycle_lambda = cycle_lambda
        self.diversity_lambda = diversity_lambda
        self.momentum_beta = momentum_beta
        self.temperature = temperature

        # Cache
        self.register_buffer(
            "interpolation_weights", torch.linspace(0, 1, num_interpolation_points)
        )
        self.register_buffer("momentum", torch.zeros(1, latent_dim))

    @torch.amp.autocast("cuda")
    def compute_adaptive_weights(self, z1, z2):
        # Thêm metric đa dạng hơn
        l2_dist = torch.norm(z1 - z2, dim=-1)
        consine_sim_F = nn.CosineSimilarity(dim=-1)
        cosine_sim = consine_sim_F(z1, z2)

        # Kết hợp nhiều metric
        similarity = (cosine_sim + 1) / 2 - torch.tanh(l2_dist)
        return torch.sigmoid(similarity / self.temperature)

    @torch.amp.autocast("cuda")
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate between two latent vectors using provided weights
        """
        weights = weights.to(z1.device)

        # Apply adaptive weighting
        adaptive_w = self.compute_adaptive_weights(z1, z2).unsqueeze(-1)
        weights = weights * adaptive_w

        return torch.lerp(z1, z2, weights)

    def linear_interpolate(
        self, z1: torch.Tensor, z2: torch.Tensor, num_points: int = None
    ) -> torch.Tensor:
        """
        Perform linear interpolation between two latent vectors
        z = α * z1 + (1-α) * z2
        """
        if num_points is None:
            num_points = self.num_points

        # Generate interpolation weights
        weights = self.interpolation_weights.view(-1, 1, 1)  # [num_points, 1, 1]

        # Compute adaptive domain weights
        domain_weights = self.compute_domain_weights(z1, z2)

        # Reshape inputs for broadcasting
        z1 = z1.unsqueeze(0)  # [1, batch_size, latent_dim]
        z2 = z2.unsqueeze(0)  # [1, batch_size, latent_dim]

        # Linear interpolation: z = α * z1 + (1-α) * z2
        interpolated = self.interpolate(z1=z1, z2=z2, weights=weights)

        """
        momentum = β * momentum + (1 - β) * (z2 - z1).mean(dim=1)
        """
        momentum = self.momentum_beta * self.momentum + (1 - self.momentum_beta) * (
            z2 - z1
        ).mean(dim=1)
        self.momentum = momentum.detach()  # Update momentum cache

        # Apply momentum correction
        correction = momentum.unsqueeze(0) * weights * (1 - weights)
        interpolated = interpolated + correction * domain_weights

        return interpolated

    @torch.amp.autocast("cuda")
    def compute_diversity_loss(self, interpolated_points: torch.Tensor) -> torch.Tensor:
        """
        Diversity loss to ensure that the latent space is diverse
        L_diversity = -Σ Σ ||z_i - z_j||₂²
        """
        # Reshape to 2D for efficient computation
        points_flat = interpolated_points.view(-1, self.latent_dim)

        # Compute pairwise distances efficiently
        norm = torch.sum(points_flat**2, dim=1, keepdim=True)
        dist_matrix = norm + norm.t() - 2 * torch.mm(points_flat, points_flat.t())

        # Get upper triangle without diagonal
        mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1)
        distances = dist_matrix * mask

        return -torch.mean(distances[distances > 0])

    @torch.amp.autocast("cuda")
    def compute_adaptive_temperature(self, z1, z2):
        """Calculate adaptive temperature based on domain similarity"""
        cosine_sim = nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(z1, z2).mean()
        return self.temperature * torch.sigmoid(similarity)

    @torch.amp.autocast("cuda")
    def compute_domain_weights(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute domain weights based on multiple similarity metrics.
        """
        # Add more metrics for better domain similarity
        l2_dist = torch.norm(z1 - z2, dim=-1)
        cosine_sim_f = nn.CosineSimilarity(dim=-1)
        cosine_sim = cosine_sim_f(z1, z2)

        # Ensure inputs are at least 2D tensors
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if z2.dim() == 1:
            z2 = z2.unsqueeze(0)

        feature_dist = torch.cdist(z1, z2, p=2).mean()

        # Combine multiple metrics
        similarity = (cosine_sim + 1.0) / 2.0 - l2_dist.tanh() - feature_dist.tanh()

        # Adaptive temperature
        temp = self.compute_adaptive_temperature(z1, z2)
        weights = torch.sigmoid(similarity / temp)

        return weights.unsqueeze(-1)

    @torch.amp.autocast("cuda")
    def compute_consistency_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, interpolated_points: torch.Tensor
    ) -> torch.Tensor:
        """
        L_consist = ||z_interp - z_target||₂²
        - Consistency loss ensures that the interpolation is consistent with the target points
        - Target points are generated by linear interpolation between z1 and z2
        - Target points mean that the interpolation should be consistent with the linear interpolation
        - Consistent interpolation is important for the generator to generate realistic images, and it helps to prevent mode collapse
        """
        # Get the number of interpolation points from interpolated_points
        num_points = interpolated_points.size(0)

        # Generate target points
        alphas = torch.linspace(0, 1, num_points, device=z1.device)
        alphas = alphas.view(-1, 1, 1)  # Shape: [num_points, 1, 1]

        # Compute domain weights
        domain_weights = self.compute_domain_weights(z1, z2)
        domain_weights = domain_weights.unsqueeze(0)  # Shape: [1, batch_size, 1]

        # Reshape inputs for broadcasting
        z1 = z1.unsqueeze(0)  # Shape: [1, batch_size, latent_dim]
        z2 = z2.unsqueeze(0)  # Shape: [1, batch_size, latent_dim]

        # Generate target points with broadcasting
        target_points = alphas * z1 + (1 - alphas) * z2  # Shape: [num_points, batch_size, latent_dim]
        target_points = target_points * domain_weights  # Apply domain weights

        # Ensure interpolated_points has the same shape as target_points
        interpolated_points = interpolated_points.view(num_points, -1, self.latent_dim)

        return F.mse_loss(interpolated_points, target_points)

    def interpolate_latents(
        self, z1: torch.Tensor, z2: torch.Tensor, num_points: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between two latent vectors with regularization
        """
        interpolated = self.linear_interpolate(z1, z2, num_points)

        if self.training:
            # Calculate losses only during training
            # smoothness_loss = self.compute_smoothness_loss(interpolated)
            consistency_loss = self.compute_consistency_loss(z1, z2, interpolated)
            diversity_loss = self.compute_diversity_loss(interpolated)
            # cycle_loss = self.compute_cycle_consistency_loss(z1, z2)
            total_loss = (
                # self.smoothness_lambda * smoothness_loss
                self.consistency_lambda * consistency_loss
                + self.diversity_lambda * diversity_loss
                # + self.cycle_lambda * cycle_loss
            )
            return interpolated, total_loss

        return interpolated, torch.tensor(0.0, device=z1.device)

    # @torch.amp.autocast("cuda")
    # def compute_cycle_consistency_loss(self, z1, z2):
    #     """Ensure cycle consistency in interpolation"""
    #     # Reshape inputs and weights for correct broadcasting
    #     z1 = z1.unsqueeze(0)  # [1, batch_size, latent_dim]
    #     z2 = z2.unsqueeze(0)  # [1, batch_size, latent_dim]
    #     weights = self.interpolation_weights.view(-1, 1, 1)  # [num_points, 1, 1]

    #     epsilon = 1e-8

    #     # Compute forward and backward interpolations
    #     forward_interp = self.interpolate(z1, z2, weights)
    #     backward_interp = self.interpolate(z2, z1, 1 - weights)

    #     cycle_loss = F.l1_loss(forward_interp, backward_interp) + epsilon
    #     return cycle_loss

    def forward(
        self, domain_latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass with gradient checkpointing
        """
        batch_size = domain_latents.size(1)
        all_interpolated = []
        total_loss = torch.tensor(0.0, device=domain_latents.device)

        # Use torch.utils.checkpoint for memory efficiency
        def create_checkpoint_function(i, j, b):
            def custom_forward(*inputs):
                z1, z2 = inputs
                interpolated = self.linear_interpolate(z1[b], z2[b])

                if self.training:
                    consistency_loss = self.compute_consistency_loss(z1[b], z2[b], interpolated)
                    diversity_loss = self.compute_diversity_loss(interpolated)
                    # cycle_loss = self.compute_cycle_consistency_loss(z1[b], z2[b])
                    loss = (
                        # self.smoothness_lambda * smoothness_loss +
                        self.consistency_lambda * consistency_loss
                        + self.diversity_lambda * diversity_loss
                        # + self.cycle_lambda * cycle_loss
                    )
                    return interpolated, loss
                return interpolated, torch.tensor(0.0, device=z1.device)
            return custom_forward

        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                if i == j:
                    continue

                z1 = domain_latents[i]
                z2 = domain_latents[j]

                for b in range(batch_size):
                    if self.training:
                        interpolated, loss = torch.utils.checkpoint.checkpoint(
                            create_checkpoint_function(i, j, b),
                            z1.detach(),
                            z2.detach(),
                            use_reentrant=True,
                        )
                    else:
                        interpolated = self.linear_interpolate(
                            z1[b : b + 1], z2[b : b + 1]
                        )
                        loss = torch.tensor(0.0, device=z1.device)

                    all_interpolated.append(interpolated)
                    total_loss += loss

        interpolated_result = torch.cat(all_interpolated, dim=0) if all_interpolated else domain_latents
        return interpolated_result, total_loss


class GLOGenerator(nn.Module):
    """Generator network following GLO paper architecture"""

    def __init__(self, latent_dim=1024, output_dim=3):
        super(GLOGenerator, self).__init__()

        # Following GLO paper architecture
        self.fc = nn.Linear(latent_dim, 14 * 14 * 256)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # size 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_dim, 4, stride=2, padding=1),  # size 112x112
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 14, 14)
        x = self.deconv(x)
        return x

class GLOModule(nn.Module):
    def __init__(
        self,
        latent_dim=1024,
        num_domains=3,
        batch_size=32,
        temperature=0.07,
        consistency_lambda=0.1,
        diversity_lambda=0.05,
        cycle_lambda=0.1,
    ):
        super(GLOModule, self).__init__()
        self.latent_dim = latent_dim
        self.num_domains = num_domains
        self.batch_size = batch_size
        self.temperature = temperature

        # Initialize components
        self.generator = GLOGenerator(latent_dim)
        self.domain_latents = nn.Parameter(
            torch.randn(num_domains, batch_size, latent_dim)
        )
        nn.init.normal_(self.domain_latents, mean=0.0, std=0.02)

        # Updated interpolator
        self.interpolator = LatentInterpolator(
            num_domains,
            latent_dim,
            consistency_lambda=consistency_lambda,
            diversity_lambda=diversity_lambda,
            cycle_lambda=cycle_lambda,
        )

        # Projection and discrimination
        self.projection_head = self.build_projection_head()

    def build_projection_head(self):
        return nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_dim, self.latent_dim // 2)),
            nn.ReLU(),
            SpectralNorm(nn.Linear(self.latent_dim // 2, self.latent_dim // 4)),
        ).to(next(self.parameters()).device)

    def get_projected_latents(self, z):
        """Project latents to lower-dimensional space for clustering"""
        return self.projection_head(z)

    def compute_clustering_loss(self, z, domain_labels, temperature=None):
        """
        Công thức Contrastive Loss chuẩn theo paper:
        L = -log[ exp(sim(z_i, z_j)/temperature) / Σ_k exp(sim(z_i, z_k)/temperature) ]
        Args:
            features: Tensor của projected features [N, D]
            domain_labels: Tensor của domain labels [N]
            temperature: Scalar cho temperature scaling
        """
        # Convert domain_labels to tensor and ensure it's 1-dimensional
        if not isinstance(domain_labels, torch.Tensor):
            domain_labels = torch.tensor(domain_labels, device=z.device)

        if domain_labels.dim() == 0:
            domain_labels = domain_labels.unsqueeze(0)

        features = self.get_projected_latents(z)

        if temperature is None:
            temperature = self.temperature

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T)  # [N, N]

        # Mask for positive pairs (same domain)
        pos_mask = (domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)).float()
        # Remove self-contrast
        pos_mask = pos_mask - torch.eye(pos_mask.shape[0], device=pos_mask.device)

        # Mask for valid positive pairs
        valid_pos_mask = (pos_mask.sum(dim=1) > 0).float()

        # Compute exp(sim/temp) for all pairs
        exp_sim = torch.exp(sim_matrix / temperature)

        # Zero out self-contrast
        exp_sim = exp_sim * (1 - torch.eye(exp_sim.shape[0], device=exp_sim.device))

        # Compute positive pairs loss
        pos_pairs = exp_sim * pos_mask
        pos_pairs = pos_pairs.sum(dim=1)  # [N]

        # Compute denominator (all possible pairs except self)
        denominator = exp_sim.sum(dim=1)  # [N]

        # Compute log ratios for valid samples
        log_ratios = -torch.log(pos_pairs / denominator + 1e-8) * valid_pos_mask

        # Normalize by number of valid positive samples
        loss = log_ratios.sum() / (valid_pos_mask.sum() + 1e-8)

        # if loss = 0, it means that all the samples are in the same domain
        return loss

    def forward(self, x: torch.Tensor, domain_idx: int) -> Tuple:
        """
        Forward pass with consistent outputs for both training and inference
        """
        batch_size = x.size(0)
        z = self.domain_latents[domain_idx, :batch_size]

        if self.training:
            # Get interpolated latents and losses
            z_interpolated, interp_loss = self.interpolator(self.domain_latents)

            # Generate images
            x_generated = self.generator(z)

            x_interpolated = self.generator(z_interpolated)

            return (
                x_generated,
                z,
                x_interpolated,
                z_interpolated,
                interp_loss
            )

        # Inference mode - generate single domain
        x_generated = self.generator(z)
        return x_generated, z, None, None, torch.tensor(0.0, device=x.device)


class CycleMixLayer(nn.Module):
    def __init__(self, hparams, device):
        super(CycleMixLayer, self).__init__()
        self.device = device
        self.sources = get_sources(hparams["dataset"], hparams["test_envs"])
        self.glo = GLOModule(
            latent_dim=hparams.get("latent_dim", 512),
            num_domains=len(self.sources),
            batch_size=hparams["batch_size"],
        ).to(device)

        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def process_domain(
        self, batch: Tuple[torch.Tensor, torch.Tensor], domain_idx: int
    ) -> Tuple:
        """
        Process a single domain batch with consistent outputs
        """
        x, y = batch

        if self.training:
            # Get all outputs during training
            x_hat, z, x_hat_interp, z_interp, interp_loss = self.glo(x, domain_idx)
            x_hat = self.norm(x_hat)
            x_hat_interp = self.norm(x_hat_interp) if x_hat_interp is not None else None

            return (x, y), (x_hat, y), (x_hat_interp, y), interp_loss

        # Simplified output for inference
        x_hat, z = self.glo(x, domain_idx)[:2]  # Only take first two outputs
        x_hat = self.norm(x_hat)
        return (x, y), (x_hat, y), (x_hat, y), torch.tensor(0.0, device=x.device)

    def forward(
        self, x: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process all domains and maintain consistent output structure
        """
        processed_domains = [
            self.process_domain(batch, idx) for idx, batch in enumerate(x)
        ]

        # Separate components
        original, generated, interpolated, losses = zip(*processed_domains)

        # Combine outputs based on training mode
        if self.training:
            all_samples = list(original) + list(generated) + list(interpolated)
        else:
            all_samples = list(original) + list(generated)

        return all_samples, (
            sum(losses) if self.training else torch.tensor(0.0, device=self.device)
        )


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(
                            bottleneck,
                            name2,
                            fuse(module2, getattr(bottleneck, bn_name)),
                        )
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(
                        bottleneck.downsample[0], bottleneck.downsample[1]
                    )
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            print("Using ResNet18")
            self.network = torchvision.models.resnet18(pretrained=False)
            self.n_outputs = 512
        else:
            print("Using ResNet50")
            self.network = torchvision.models.resnet50(pretrained=False)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


class ViTFeaturizer(torch.nn.Module):
    """ViT feature extractor with consistent output dimension"""

    def __init__(self, input_shape, hparams):
        super(ViTFeaturizer, self).__init__()
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # Match output dimension with original ResNet
        if hparams.get("resnet18", True):
            self.n_outputs = 512
        else:
            self.n_outputs = 2048

        # Add linear projection to match expected feature dimension
        self.feature_projection = nn.Linear(768, self.n_outputs)

        self.dropout = nn.Dropout(hparams.get("vit_dropout", 0.1))
        self.hparams = hparams

    def forward(self, x):
        """
        Input: (batch_size, channels, height, width)
        Output: (batch_size, n_outputs) where n_outputs matches ResNet
        """
        # ViT forward pass
        outputs = self.vit(x)
        # Get CLS token features
        features = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, 768)
        # Project to match ResNet dimension
        features = self.feature_projection(features)  # Shape: (batch_size, n_outputs)
        return self.dropout(features)

    def train(self, mode=True):
        super().train(mode)
        # Keep batch norm in eval mode if it exists
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


from torch.utils.model_zoo import load_url


class ResNetMoCo(torch.nn.Module):
    """ResNet with MoCo-v2 pretraining, with frozen BatchNorm"""

    def __init__(self, input_shape, hparams):
        super(ResNetMoCo, self).__init__()

        # MoCo v2 checkpoint URL
        moco_v2_path = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"

        if hparams["resnet18"]:
            print("Using ResNet18")
            self.network = torchvision.models.resnet18(pretrained=False)
            self.n_outputs = 512
        else:
            print("Using ResNet50 with MoCo-v2 weights")
            # Khởi tạo model ResNet50 base
            self.network = torchvision.models.resnet50(pretrained=False)
            self.n_outputs = 2048

            try:
                # Tải MoCo v2 checkpoint
                checkpoint = load_url(moco_v2_path, progress=True)

                # Xử lý state dict từ MoCo v2
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for k in list(state_dict.keys()):
                    # Loại bỏ prefix 'module.encoder_q.'
                    if k.startswith("module.encoder_q."):
                        new_state_dict[k[len("module.encoder_q.") :]] = state_dict[k]

                # Tải weights vào model
                msg = self.network.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded MoCo v2 weights. Missing keys: {msg.missing_keys}")

            except Exception as e:
                print(f"Error loading MoCo v2 weights: {str(e)}")
                print("Falling back to random initialization")

        # Xử lý số kênh đầu vào
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()
            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # Loại bỏ fully connected layer
        del self.network.fc
        self.network.fc = Identity()

        # Thêm batch norm freeze và dropout
        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters"""
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    print(input_shape)
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNetMoCo(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features),
        )
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs, num_classes, hparams["nonlinear_classifier"]
        )
        self.net = nn.Sequential(featurizer, classifier)
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
