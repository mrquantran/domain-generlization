# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from transformers import ViTModel

import numpy as np

from domainbed.lib import wide_resnet
import copy

# CYCLEGAN Experiments
from domainbed.cyclegan.utils import get_sources

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import wasserstein_distance
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm


import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import silhouette_score
import numpy.polynomial.polynomial as poly


class ALDModule(nn.Module):
    """Adaptive Latent Dimensionality Module optimized for domain generalization"""

    def __init__(
        self,
        initial_dim=512,  # Increased from 256 to capture more domain-invariant features initially
        latent_decrease=5,  # More conservative decrease to preserve important features
        patience=5,  # Reduced to check more frequently
        window_size=20,  # Keep as per paper
        n_classes=7,  # Depends on dataset (7 for PACS, 65 for OfficeHome)
        min_dim=64,  # Increased minimum to ensure enough capacity for cross-domain features
        slope_threshold=5e-4,  # More sensitive threshold
        improvement_threshold=1e-4,  # New parameter for meaningful improvements
    ):
        super(ALDModule, self).__init__()
        self.current_dim = initial_dim
        self.latent_decrease = latent_decrease
        self.patience = patience
        self.window_size = window_size
        self.n_classes = n_classes
        self.min_dim = min_dim
        self.slope_threshold = slope_threshold
        self.improvement_threshold = improvement_threshold

        # Enhanced tracking
        self.s_scores = []
        self.recon_losses = []
        self.compressing = True
        self.best_score = float("-inf")
        self.epochs_no_improve = 0
        self.slow_compression = False

    def polyfit_slope(self, values, deg=1):
        """Calculate slope using polynomial fitting with enhanced stability"""
        if len(values) < self.window_size:
            return 0.0

        y = np.array(values[-self.window_size :])
        x = np.arange(len(y))

        # Normalize values for numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) != 0 else 1
        y_normalized = (y - y_mean) / y_std

        try:
            coeffs = poly.polyfit(x, y_normalized, deg)
            slope = coeffs[1] * y_std
            return slope
        except np.linalg.LinAlgError:
            return 0.0

    def should_adjust_dimension(self, epoch) -> bool:
        """Enhanced dimension adjustment check with domain generalization focus"""
        if self.current_dim <= self.min_dim:
            return False
            
        if epoch % self.patience == 0:
            # Step 16-19: Calculate slopes and check for improvements
            recon_slope = self.polyfit_slope(self.recon_losses)
            sil_slope = self.polyfit_slope(self.s_scores)

            # Check if all slopes are positive
            if recon_slope > 0 and sil_slope > 0:
                self.compressing = False
                return False

            # Transition to slower compression if silhouette score slope is positive
            if sil_slope > 0 and self.latent_decrease > 1:
                self.latent_decrease = max(1, self.latent_decrease // 2)
                
        return True

    def adjust_dimension(self):
        """Adjust dimension with domain generalization considerations"""
        if not self.compressing:
            return False, self.current_dim

        # Calculate new dimension
        decrease = self.latent_decrease
        if self.current_dim - decrease < self.min_dim:
            decrease = self.current_dim - self.min_dim

        new_dim = max(self.min_dim, self.current_dim - decrease)

        # Stop if we can't decrease further
        if new_dim == self.current_dim:
            self.compressing = False
            return False, self.current_dim

        self.current_dim = new_dim
        return True, new_dim

    def update_metrics(self, x_source_list, x_target_list):
        """Update metrics with domain-aware calculations"""
        reconstruction_losses = []
        silhouette_scores = []

        # Process each domain pair
        for (x_source, _), (x_target, _) in zip(x_source_list, x_target_list):
            if not torch.is_tensor(x_source) or not torch.is_tensor(x_target):
                raise ValueError(
                    f"Invalid input types: {type(x_source)}, {type(x_target)}"
                )

            # Calculate domain-aware silhouette score
            z_numpy = x_source.view(x_source.shape[0], -1).detach().cpu().numpy()
            if z_numpy.shape[0] < self.n_classes:
                continue  # Skip if batch is too small instead of raising error

            kmeans = KMeans(n_clusters=self.n_classes, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(z_numpy)

            if len(np.unique(labels)) > 1:
                s_score = silhouette_score(z_numpy, labels, metric="euclidean")
                silhouette_scores.append(float(s_score))

            # Calculate reconstruction loss with domain alignment consideration
            recon_loss = F.mse_loss(x_source, x_target).item()
            reconstruction_losses.append(recon_loss)
            # Calculate reconstruction loss using KL divergence
            mu_source, logvar_source = torch.mean(x_source, dim=0), torch.var(
                x_source, dim=0
            )
            mu_target, logvar_target = torch.mean(x_target, dim=0), torch.var(
                x_target, dim=0
            )

            # Ensure numerical stability
            logvar_source = torch.clamp(logvar_source, min=1e-6)
            logvar_target = torch.clamp(logvar_target, min=1e-6)

            # KL divergence between source and target distributions
            kl_div = 0.5 * torch.sum(
                logvar_target
                - logvar_source
                + (logvar_source.exp() + (mu_source - mu_target).pow(2))
                / logvar_target.exp()
                - 1
            )
            reconstruction_losses.append(kl_div.item())

        # Update tracking metrics if we have valid scores
        if silhouette_scores and reconstruction_losses:
            self.s_scores.append(torch.tensor(silhouette_scores).mean())
            self.recon_losses.append(torch.tensor(reconstruction_losses).mean())

        return bool(silhouette_scores and reconstruction_losses)


class GLOGenerator(nn.Module):
    """Generator network with proper dimension handling"""

    def __init__(self, latent_dim=256, output_shape=(3, 224, 224)):
        super(GLOGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        # Calculate proper dimensions
        c, h, w = output_shape
        self.init_h = h // 16
        self.init_w = w // 16
        self.init_channels = 256

        # Proper FC layer initialization
        self.fc = nn.Linear(latent_dim, self.init_h * self.init_w * self.init_channels)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, c, 4, stride=2, padding=1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        batch_size = z.size(0)

        # Ensure proper shape handling
        x = self.fc(z)
        x = x.view(batch_size, self.init_channels, self.init_h, self.init_w)
        x = self.deconv(x)
        return x


class GLOModule(nn.Module):
    def __init__(
        self, latent_dim=512, num_domains=3, num_classes=7, batch_size=32, device="cuda"
    ):
        super(GLOModule, self).__init__()
        self.device = device
        self.ald = ALDModule(initial_dim=latent_dim, n_classes=num_classes).to(
            self.device
        )

        # Add dimension validator
        self.input_validator = nn.Linear(self.ald.current_dim, self.ald.current_dim).to(
            self.device
        )

        # Initialize with proper output shape
        self.generator = GLOGenerator(
            latent_dim=self.ald.current_dim,
            output_shape=(3, 224, 224),
        ).to(self.device)

        # Add dimension adapter layer
        self.fc_output_size = (
            self.generator.init_h * self.generator.init_w * self.generator.init_channels
        )
        self.dim_adapter = nn.ModuleDict(
            {
                "fc": nn.Linear(self.ald.current_dim, self.fc_output_size),
                "relu": nn.ReLU(),
                "bn": nn.BatchNorm1d(self.fc_output_size),
            }
        ).to(device)

        self.domain_latents = nn.Parameter(
            torch.randn(
                num_domains, batch_size, self.ald.current_dim, device=self.device
            )
        )
        self._init_domain_latents()

    def _init_domain_latents(self):
        nn.init.normal_(self.domain_latents, mean=0.0, std=0.02)

    def update_dimension(
        self,
        x_source=None,
        x_target=None,
        current_epoch=0,
    ):
        old_dim = self.ald.current_dim
        self.ald.update_metrics(x_source_list=x_source, x_target_list=x_target)

        action = self.ald.should_adjust_dimension(current_epoch)

        if action:
            action, new_dim = self.ald.adjust_dimension()
            print(f"Dimension adjusted: {old_dim} -> {new_dim}")
            if new_dim != old_dim:
                # Update domain latents
                new_domain_latents = torch.randn(
                    self.domain_latents.size(0),
                    self.domain_latents.size(1),
                    new_dim,
                    device=self.device,
                    dtype=self.domain_latents.dtype,
                )

                # Copy old values for stable transition
                min_dim = min(old_dim, new_dim)
                new_domain_latents[:, :, :min_dim] = self.domain_latents[:, :, :min_dim]
                self.domain_latents = nn.Parameter(new_domain_latents)

                # Update input validator
                self.input_validator = nn.Linear(new_dim, new_dim).to(self.device)

                # Update dimension adapter
                self.fc_output_size = (
                    self.generator.init_h
                    * self.generator.init_w
                    * self.generator.init_channels
                )
                self.dim_adapter = nn.ModuleDict(
                    {
                        "fc": nn.Linear(new_dim, self.fc_output_size),
                        "relu": nn.ReLU(),
                        "bn": nn.BatchNorm1d(self.fc_output_size),
                    }
                ).to(self.device)

                self.reset_batch_norm()
                self.validate_dimensions()

    def reset_batch_norm(self):
        """Reset BatchNorm statistics when dimension changes"""
        if hasattr(self.dim_adapter, "bn"):
            self.dim_adapter.bn.reset_parameters()
            self.dim_adapter.bn.to(self.device)

    def validate_dimensions(self):
        current_dim = self.ald.current_dim

        assert self.domain_latents.size(-1) == current_dim
        assert self.input_validator.in_features == current_dim
        assert self.dim_adapter["fc"].in_features == current_dim

    def forward(self, x, domain_idx):
        device = x.device
        batch_size = x.size(0)

        # Get latent vectors with correct dimension
        z = self.domain_latents[domain_idx, :batch_size].to(device)
        current_dim = self.ald.current_dim

        if z.size(-1) != current_dim:
            # Handle dimension mismatch
            new_z = torch.zeros(batch_size, current_dim, device=device)
            min_dim = min(z.size(-1), current_dim)
            new_z[:, :min_dim] = z[:, :min_dim]
            z = new_z

        z = self.input_validator(z)
        z = self.dim_adapter["fc"](z)
        z = self.dim_adapter["relu"](z)
        z = self.dim_adapter["bn"](z)

        z = z.view(
            batch_size,
            self.generator.init_channels,
            self.generator.init_h,
            self.generator.init_w,
        )

        generated = self.generator.deconv(z)
        return generated, z.view(batch_size, -1)


class ProjectionHead(nn.Module):
    """Enhanced projection head with dynamic dimension handling"""

    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()

        # Adjust hidden dimension based on input
        adjusted_hidden_dim = min(hidden_dim, input_dim * 2)

        self.net = nn.Sequential(
            nn.Linear(input_dim, adjusted_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(adjusted_hidden_dim),
            nn.Linear(adjusted_hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning"""

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(-1, 1)
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask.fill_diagonal_(0)

        negative_samples = sim[mask].reshape(2 * batch_size, -1)

        labels = torch.zeros(2 * batch_size).long().to(positive_samples.device)
        logits = torch.cat([positive_samples, negative_samples], dim=1)

        loss = self.criterion(logits, labels)
        return loss


class CycleMixLayer(nn.Module):
    def __init__(self, hparams, device):
        super(CycleMixLayer, self).__init__()
        self.device = device
        self.sources = get_sources(hparams["dataset"], hparams["test_envs"])
        self.num_classes = len(self.sources)
        self.feature_dim = 512 if hparams.get("resnet18", True) else 2048

        self.latent_dim = hparams.get("latent_dim", 512)

        # GLO components
        self.glo = GLOModule(
            latent_dim=self.latent_dim,
            num_domains=len(self.sources),
            batch_size=hparams["batch_size"],
            device=self.device,
            num_classes=self.num_classes,
        ).to(device)

        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            input_dim=self.latent_dim,  # Use latent dimension
            hidden_dim=hparams.get("proj_hidden_dim", 2048),
            output_dim=hparams.get("proj_output_dim", 128),
        ).to(device)

        # Contrastive loss
        self.nt_xent = NTXentLoss(temperature=hparams.get("temperature", 0.5))

        # Image normalization
        self.norm = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def process_domain(self, batch, domain_idx, featurizer):
        x, y = batch
        device = next(featurizer.parameters()).device
        x = x.to(device)
        y = y.to(device)

        # Create adapter if not exists
        if not hasattr(self, "featurizer_adapter"):
            self.featurizer_adapter = FeaturizerAdapter(
                featurizer, self.glo.ald.current_dim, device
            ).to(device)

        # Generate domain-mixed samples
        x_hat, z = self.glo(x, domain_idx)

        # Extract and project features using adapter
        feat = self.featurizer_adapter(x)
        proj = self.projection(feat)
        proj = F.normalize(proj, dim=1)

        # Normalize generated samples
        x_hat = x_hat.to(device)
        x_hat = self.norm(x_hat)

        return (x, y), (x_hat, y), proj

    def forward(self, x: list, featurizer):
        """
        Args:
            x: list of domain batches [(x_1, y_1), ..., (x_n, y_n)]
            featurizer: Feature extractor from DomainBed
        Returns:
            original_and_generated: list of tuples [(x_i, y_i), (x_i_hat, y_i)]
            projections: list of normalized projections [proj_1, ..., proj_n]of normalized projections [proj_1, ..., proj_n]
        """
        num_domains = len(x)

        # Process each domain
        processed_domains = [
            self.process_domain(batch, idx, featurizer) for idx, batch in enumerate(x)
        ]

        # Unzip results
        original_samples, generated_samples, projections = zip(*processed_domains)

        # Combine original and generated samples
        all_samples = list(original_samples) + list(generated_samples)

        # Return in required format
        return all_samples, [projections]


class FeaturizerAdapter(nn.Module):
    def __init__(self, featurizer, latent_dim, device="cuda"):
        super(FeaturizerAdapter, self).__init__()
        self.device = device
        self.featurizer = featurizer
        self.feature_dim = self._get_feature_dim()

        # Two-stage adaptation for better dimension handling
        intermediate_dim = min(self.feature_dim, 1024)  # Prevent excessive dimensions
        self.adapter = nn.Sequential(
            nn.Linear(self.feature_dim, intermediate_dim),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim),
            nn.Linear(intermediate_dim, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
        ).to(device)

    def _get_feature_dim(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            features = self.featurizer(dummy_input)
            return features.shape[1]

    def forward(self, x):
        features = self.featurizer(x)
        features = features.view(features.size(0), -1)  # Flatten if needed
        adapted_features = self.adapter(features)
        return adapted_features


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
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
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
        return ViTFeaturizer(input_shape, hparams)
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
