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
        latent_decrease=4,  # More conservative decrease to preserve important features
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

        return False

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
    """Generator network following GLO paper architecture"""

    def __init__(self, latent_dim=512, output_dim=3):
        super(GLOGenerator, self).__init__()

        # Following GLO paper architecture
        self.fc = nn.Linear(latent_dim, 14 * 14 * 256)

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
            nn.ConvTranspose2d(32, output_dim, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 14, 14)
        x = self.deconv(x)
        return x

class GLOModule(nn.Module):
    def __init__(self, latent_dim=512, num_domains=3, batch_size=32):
        super(GLOModule, self).__init__()
        self.latent_dim = latent_dim
        self.num_domains = num_domains
        self.batch_size = batch_size

        # Generator following GLO paper
        self.generator = GLOGenerator(latent_dim)

        # Learnable latent codes for each domain - key difference from original GLO
        # We maintain separate latent codes for each domain
        self.domain_latents = nn.Parameter(
            torch.randn(num_domains, batch_size, latent_dim)
        )

        # Initialize with normal distribution as per GLO paper
        nn.init.normal_(self.domain_latents, mean=0.0, std=0.02)

    def forward(self, x, domain_idx):
        batch_size = x.size(0)

        # Get corresponding latent codes for the domain
        z = self.domain_latents[domain_idx, :batch_size]

        # Generate images
        generated = self.generator(z)
        return generated, z


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
        # Dynamic feature dimension based on backbone
        self.feature_dim = 512 if hparams.get("resnet18", True) else 2048

        # GLO components
        self.glo = GLOModule(
            latent_dim=hparams.get("latent_dim", 512),
            num_domains=len(self.sources),
            batch_size=hparams["batch_size"],
        ).to(device)

        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            input_dim=self.feature_dim,
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
        """Process single domain data"""
        x, y = batch

        # Generate domain-mixed samples
        x_hat, z = self.glo(x, domain_idx)

        # Extract and project features
        feat = featurizer(x)
        proj = self.projection(feat)
        proj = F.normalize(proj, dim=1)

        # Normalize generated samples
        x_hat = self.norm(x_hat)

        return (x, y), (x_hat, y), proj

    def forward(self, x: list, featurizer):
        """
        Args:
            x: list of domain batches [(x_1, y_1), ..., (x_n, y_n)]
            featurizer: Feature extractor from DomainBed
        Returns:
            original_and_generated: list of tuples [(x_i, y_i), (x_i_hat, y_i)]
            projections: list of normalized projections [proj_1, ..., proj_n]
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
