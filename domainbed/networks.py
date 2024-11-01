# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple
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

import torch
import torch.nn as nn
import numpy as np

from sklearn.decomposition import PCA

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
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # size 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_dim, 4, stride=2, padding=1), # size 112x112
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 14, 14)
        x = self.deconv(x)
        return x

class GLOModule(nn.Module):

    def __init__(self, latent_dim=1024, num_domains=3, batch_size=32, temperature=0.07):
        super(GLOModule, self).__init__()
        print('latent_dim:', latent_dim)
        self.latent_dim = latent_dim
        self.num_domains = num_domains
        self.batch_size = batch_size
        self.temperature = temperature

        # Generator following GLO paper
        self.generator = GLOGenerator(latent_dim)

        # Learnable latent codes for each domain - key difference from original GLO
        # We maintain separate latent codes for each domain
        self.encoder = Encoder(latent_dim)

        # Initialize with normal distribution as per GLO paper
        nn.init.normal_(self.domain_latents, mean=0.0, std=0.02)

        # Projection head for clustering
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
        )

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

        return loss # if loss = 0, it means that all the samples are in the same domain

    def sample_latents(self, domain_idx, batch_size):
        if self.training:
            mean = self.domain_latents.mean[domain_idx]
            std = self.domain_latents.std[domain_idx]
            z = torch.normal(mean, std, size=(batch_size, self.latent_dim))
        else:
            z = self.domain_latents[domain_idx, :batch_size]
        return z

    def forward(self, x, domain_idx):
        batch_size = x.size(0)

        # Encode input images
        mu, logvar = self.encoder(x)

        # Get corresponding latent codes for the domain
        z = self.sample_latents(domain_idx, batch_size)

        # Generate images
        generated = self.generator(z)
        return generated, z

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # Khởi tạo mô hình EfficientNet-B1 mà không sử dụng pretrained weights
        self.efficientnet = torchvision.models.efficientnet_b1(pretrained=torchvision.models.EfficientNet_B1_Weights.DEFAULT)

        # Lấy số features từ lớp cuối cùng của EfficientNet-B1
        in_features = self.efficientnet.classifier[1].in_features

        # Mean (mu) and log-variance (logvar) layers
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        features = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(features)
        x = torch.flatten(x, 1)

        # Compute mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class CycleMixLayer(nn.Module):
    def __init__(self, hparams, device):
        super(CycleMixLayer, self).__init__()
        self.device = device
        self.sources = get_sources(hparams["dataset"], hparams["test_envs"])
        # Dynamic feature dimension based on backbone
        self.feature_dim = 512 if hparams.get("resnet18", True) else 2048

        # GLO components
        self.glo = GLOModule(
            latent_dim=hparams.get("latent_dim", 512),
            num_domains=len(self.sources),
            batch_size=hparams["batch_size"],
        ).to(device)

        # Image normalization
        self.norm = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        num_interpolation_points = 5
        self.num_points = num_interpolation_points
        self.register_buffer(
            "interpolation_weights", torch.linspace(0, 1, num_interpolation_points)
        )

    def process_domain(self, batch, domain_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process single domain data"""
        x, y = batch

        # Generate domain-mixed samples
        x_hat, z = self.glo(x, domain_idx)

        # Normalize generated samples
        x_hat = self.norm(x_hat)

        return (x, y), (x_hat, y), z

    @torch.amp.autocast("cuda")
    def interpolate(
        self, z1: torch.Tensor, z2: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate between two latent vectors using provided weights
        """
        weights = weights.to(z1.device)

        return torch.lerp(z1, z2, weights)

    def linear_interpolate(self, z1: torch.Tensor, z2: torch.Tensor, num_points: int):
        """
        Perform linear interpolation between two latent vectors
        z = α * z1 + (1-α) * z2
        """
        if num_points is None:
            num_points = self.num_points

        # Generate interpolation weights
        weights = self.interpolation_weights.view(-1, 1, 1)  # [num_points, 1, 1]

        # Reshape inputs for broadcasting
        z1 = z1.unsqueeze(0)  # [1, batch_size, latent_dim]
        z2 = z2.unsqueeze(0)  # [1, batch_size, latent_dim]

        # Linear interpolation: z = α * z1 + (1-α) * z2
        interpolated = self.interpolate(z1=z1, z2=z2, weights=weights)

        return interpolated

    def forward(self, x: list):
        """
        Args:
            x: list of domain batches [(x_1, y_1), ..., (x_n, y_n)]
            featurizer: Feature extractor from DomainBed
        Returns:
            original_and_generated: list of tuples [(x_i, y_i), (x_i_hat, y_i)]
        """
        num_domains = len(x)

        # Process each domain
        processed_domains = [
            self.process_domain(batch, idx) for idx, batch in enumerate(x)
        ]

        generated_interpolated_samples = []
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                if i == j:
                    continue

                # Get latent codes for each domain
                _, _, z1 = processed_domains[i]
                _, _, z2 = processed_domains[j]

                # Linear interpolation
                interpolated = self.linear_interpolate(z1, z2, num_points=5)

                # Generate samples
                generated_interpolated = self.glo.generator(interpolated)

                generated_interpolated_samples.append(generated_interpolated)

        # Unzip results
        original_samples, generated_samples, z_samples = zip(*processed_domains)

        # Combine original and generated samples
        all_samples = list(original_samples) + list(generated_samples)

        # Return in required format
        return all_samples

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
            print('Using ResNet18')
            self.network = torchvision.models.resnet18(pretrained=False)
            self.n_outputs = 512
        else:
            print('Using ResNet50')
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
