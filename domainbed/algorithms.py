# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from sklearn.metrics import silhouette_score
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
import torchvision
from torchvision.transforms import v2
from torchvision.utils import save_image
import copy
import numpy as np
from collections import OrderedDict
import os
from torch.utils.model_zoo import load_url
import math

try:
    from backpack import backpack, extend  # type: ignore
    from backpack.extensions import BatchGrad  # type: ignore
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches,
    split_meta_train_test,
    ParamDict,
    MovingAverage,
    l2_between_dicts,
    proj,
    Nonparametric,
)

ALGORITHMS = [
    "ERM",
    "CUTOUT",
    "CUTMIX",
    "CYCLEMIX",
    "Fish",
    "IRM",
    "GroupDRO",
    "Mixup",
    "MLDG",
    "CORAL",
    "MMD",
    "DANN",
    "CDANN",
    "MTL",
    "SagNet",
    "ARM",
    "VREx",
    "RSC",
    "SD",
    "ANDMask",
    "SANDMask",
    "IGA",
    "SelfReg",
    "Fishr",
    "TRM",
    "IB_ERM",
    "IB_IRM",
    "CAD",
    "CondCAD",
    "Transfer",
    "CausIRL_CORAL",
    "CausIRL_MMD",
    "EQRM",
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class CUTOUT(Algorithm):
    """
    Empirical Risk Minimization (CUTOUT)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CUTOUT, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class CUTMIX(Algorithm):
    """
    Empirical Risk Minimization with CutMix (CUTMIX)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CUTMIX, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.cutmix = v2.CutMix(num_classes=num_classes)

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        all_x, all_y = self.cutmix(all_x, all_y)

        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class Identity(nn.Module):
    """Identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def save_images(x, x_generated, current_epoch):
    """
    Save the original and generated images to file path.
    """
    os.makedirs("generated_images", exist_ok=True)
    print(f"Image shape: {x.shape}")
    print(f"Generated image shape: {x_generated.shape}")

    # Ensure images are in RGB format
    if x.shape[1] == 1:  # If grayscale, convert to RGB
        x = x.repeat(1, 3, 1, 1)

    if x_generated.shape[1] == 1:  # If grayscale, convert to RGB
        x_generated = x_generated.repeat(1, 3, 1, 1)

    # Save original images
    original_img_path = f"generated_images/original_{current_epoch}.png"
    save_image(x, original_img_path, nrow=8, normalize=True)
    print(f"Original images saved to {original_img_path}")

    # Scale generated images from [-1, 1] to [0, 1]
    images = (x_generated.detach().cpu() + 1) / 2

    # Save generated images
    generated_img_path = f"generated_images/generated_{current_epoch}.png"
    save_image(images, generated_img_path, nrow=8, normalize=True)
    print(f"Generated images saved to {generated_img_path}")

# class VAEEncoder(nn.Module):
#     """VAE Encoder using ResNet50 with ImageNet pretraining"""

#     def __init__(self, input_shape, latent_dim):
#         super(VAEEncoder, self).__init__()

#         # Load pretrained ResNet50
#         self.backbone = torchvision.models.resnet50(
#             weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
#         )
#         backbone_dim = 2048

#         # Handle input channels
#         nc = input_shape[0]
#         if nc != 3:
#             tmp = self.backbone.conv1.weight.data.clone()
#             self.backbone.conv1 = nn.Conv2d(
#                 nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#             )
#             for i in range(nc):
#                 self.backbone.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

#         # Remove FC layer
#         self.backbone.fc = Identity()

#         # VAE heads
#         self.fc_mu = nn.Linear(backbone_dim, latent_dim)
#         self.fc_logvar = nn.Linear(backbone_dim, latent_dim)

#         self._freeze_bn()

#     def _freeze_bn(self):
#         """Freeze BatchNorm layers"""
#         for m in self.backbone.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def forward(self, x):
#         features = self.backbone(x)
#         return self.fc_mu(features), self.fc_logvar(features)

#     def train(self, mode=True):
#         """Override train mode to keep BN frozen"""
#         super().train(mode)
#         self._freeze_bn()
#         return self


class VAEEncoder(nn.Module):
    """VAE Encoder using ResNet with MoCo-v2 pretraining"""

    def __init__(self, input_shape, latent_dim):
        super(VAEEncoder, self).__init__()

        # MoCo v2 checkpoint URL
        self.moco_v2_path = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"

        self.backbone = torchvision.models.resnet50(pretrained=False)
        backbone_dim = 2048
        self._load_moco_weights()

        # Handle input channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.backbone.conv1.weight.data.clone()
            self.backbone.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            for i in range(nc):
                self.backbone.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # Remove FC layer
        self.backbone.fc = Identity()

        # VAE heads
        self.fc_mu = nn.Linear(backbone_dim, latent_dim)
        self.fc_logvar = nn.Linear(backbone_dim, latent_dim)

        self._freeze_bn()

    def _load_moco_weights(self):
        """Load MoCo v2 pretrained weights"""
        try:
            checkpoint = load_url(self.moco_v2_path, progress=True)
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}

            for k in list(state_dict.keys()):
                if k.startswith("module.encoder_q."):
                    new_state_dict[k[len("module.encoder_q.") :]] = state_dict[k]

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded MoCo v2 weights. Missing keys: {msg.missing_keys}")

        except Exception as e:
            print(f"Error loading MoCo v2 weights: {str(e)}")
            print("Falling back to random initialization")

    def _freeze_bn(self):
        """Freeze BatchNorm layers"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_mu(features), self.fc_logvar(features)

    def train(self, mode=True):
        """Override train mode to keep BN frozen"""
        super().train(mode)
        self._freeze_bn()

class AdaptiveDomainNorm(nn.Module):
    """Enhanced normalization with batch statistics and domain-specific parameters"""
    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = momentum

        # Domain-specific layers
        self.norm = nn.ModuleList(
            [nn.LayerNorm(num_features) for _ in range(num_domains)]
        )

        # Running statistics for each domain
        self.register_buffer("running_mean", torch.zeros(num_domains, num_features))
        self.register_buffer("running_var", torch.ones(num_domains, num_features))
        self.register_buffer("num_batches_tracked", torch.zeros(num_domains))

        # Learnable parameters
        self.domain_gates = nn.Parameter(torch.ones(num_domains) / num_domains)
        self.scale = nn.Parameter(torch.ones(num_domains, num_features))
        self.bias = nn.Parameter(torch.zeros(num_domains, num_features))

        # Instance normalization for unknown domains
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=False)

    def _update_stats(self, x, domain_idx):
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased=False)

                # Update running stats
                if self.num_batches_tracked[domain_idx] == 0:
                    self.running_mean[domain_idx] = batch_mean
                    self.running_var[domain_idx] = batch_var
                else:
                    self.running_mean[domain_idx] = (
                        1 - self.momentum
                    ) * self.running_mean[domain_idx] + self.momentum * batch_mean
                    self.running_var[domain_idx] = (
                        1 - self.momentum
                    ) * self.running_var[domain_idx] + self.momentum * batch_var
                self.num_batches_tracked[domain_idx] += 1

    def forward(self, x, domain_idx=None):
        if domain_idx is not None:
            # Update statistics
            self._update_stats(x, domain_idx)

            # Normalize with domain-specific stats
            if self.training:
                mean = x.mean(0)
                var = x.var(0, unbiased=False)
            else:
                mean = self.running_mean[domain_idx]
                var = self.running_var[domain_idx]

            normalized = (x - mean) / torch.sqrt(var + self.eps)
            normalized = self.norm[domain_idx](normalized)
            return self.scale[domain_idx] * normalized + self.bias[domain_idx]

        inst_norm = self.instance_norm(x.unsqueeze(1)).squeeze(1)
        gates = F.softmax(self.domain_gates, dim=0)

        out = 0
        for i in range(self.num_domains):
            domain_norm = self.norm[i](inst_norm)
            out += gates[i] * (self.scale[i] * domain_norm + self.bias[i])
        return out

class MultiDomainVAEEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim, num_domains):
        super(MultiDomainVAEEncoder, self).__init__()

        self.num_domains = num_domains
        self.latent_dim = latent_dim
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.domain_gates = nn.Parameter(torch.ones(num_domains) / num_domains)

        # Number of attention heads
        self.n_heads = 4
        self.head_dim = latent_dim // self.n_heads
        assert latent_dim % self.n_heads == 0, "latent_dim must be divisible by n_heads"

        # Domain encoders and norm
        self.domain_encoders = nn.ModuleList(
            [VAEEncoder(input_shape, latent_dim) for _ in range(num_domains)]
        )
        self.domain_norm = AdaptiveDomainNorm(latent_dim, num_domains)

        # Multi-head attention components
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.o_proj = nn.Linear(latent_dim, latent_dim)

        # Layer norm for attention
        self.attention_norm = nn.LayerNorm(latent_dim)

        # Hierarchical mixing with attention
        self.hierarchical_level1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            ) for _ in range(num_domains)
        ])

    def attention(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Split into heads
        query = query.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights
        context = torch.matmul(attention_weights, value)

        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)

        return self.o_proj(context)

    def forward(self, x):
        batch_size = x.shape[0]

        # Get encodings from domain encoders
        all_mus = []
        all_logvars = []
        for i, encoder in enumerate(self.domain_encoders):
            mu, logvar = encoder(x)
            mu = self.domain_norm(mu, i)
            all_mus.append(mu)
            all_logvars.append(logvar)

        all_mus = torch.stack(all_mus, dim=1)  # [batch_size, num_domains, latent_dim]
        all_logvars = torch.stack(all_logvars, dim=1)

        # Apply multi-head self-attention
        q = self.q_proj(all_mus)
        k = self.k_proj(all_mus)
        v = self.v_proj(all_mus)

        attended_features = self.attention(q, k, v)
        attended_features = self.attention_norm(attended_features + all_mus)  # Add residual

        # Level 1 mixing with attended features
        level1_features = []
        for i in range(self.num_domains):
            left_idx = (i - 1) % self.num_domains
            right_idx = (i + 1) % self.num_domains

            # Use attended features for mixing
            neighbor_features = torch.cat([
                attended_features[:, left_idx, :],
                attended_features[:, right_idx, :]
            ], dim=1)

            level1_mixed = self.hierarchical_level1[i](neighbor_features)
            level1_mixed = level1_mixed + neighbor_features[:, :self.latent_dim]  # Residual
            level1_features.append(level1_mixed)

        # Global mixing with dynamic temperature-scaled gating
        gates = F.softmax(self.domain_gates / self.temperature, dim=0)
        mixed_features = []
        for i, feat in enumerate(level1_features):
            mixed_features.append(feat * gates[i])

        mixed_z = sum(mixed_features)
        mixed_mu = self.domain_norm(mixed_z, None)
        mixed_logvar = torch.zeros_like(mixed_mu)

        return mixed_z, mixed_mu, mixed_logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape, h_dim, w_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_shape = output_shape

        # Calculate starting dimensions
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.start_channels = 512

        # Project and reshape
        self.fc = nn.Linear(latent_dim, self.start_channels * self.h_dim * self.w_dim)

        # Decoder layers - careful to match input dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.start_channels, 256,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, output_shape[0],
                kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.start_channels, self.h_dim, self.w_dim)
        # Decode
        x = self.decoder(x)
        return x

class CYCLEMIX(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CYCLEMIX, self).__init__(input_shape, num_classes, num_domains, hparams)

        # Giữ nguyên các hyperparameters hiện tại
        self.input_shape = input_shape
        self.latent_dim = hparams.get("latent_dim", 512)
        self.beta = hparams.get("beta", 4.0)
        self.grad_clip = hparams.get("grad_clip", 1.0)

        # Spatial dimensions
        self.h_dim = input_shape[1] // 16
        self.w_dim = input_shape[2] // 16

        # Encoder với hierarchical mixing
        self.encoder = MultiDomainVAEEncoder(
            input_shape=input_shape,
            latent_dim=self.latent_dim,
            num_domains=num_domains
        )

        # Giữ nguyên các components khác
        self.decoder = VAEDecoder(
            latent_dim=self.latent_dim,
            output_shape=input_shape,
            h_dim=self.h_dim,
            w_dim=self.w_dim
        )

        self.domain_embeddings = nn.Parameter(
            torch.randn(num_domains, self.latent_dim)
        )

        self.classifier = nn.Sequential(
            nn.Unflatten(1, (self.latent_dim, 1)),
            nn.Conv1d(self.latent_dim, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Optimizer và scheduling giữ nguyên
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifier.parameters()) +
            [self.domain_embeddings],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )

        steps_per_epoch = hparams["steps_per_epoch"]
        num_epochs = hparams["num_epochs"]
        total_steps = int(steps_per_epoch * num_epochs)

        self.total_steps = total_steps

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=hparams["lr"] * 10,
            total_steps=self.total_steps,
            three_phase=False,
        )

        self.current_epoch = 0
        self.grad_scaler = torch.amp.GradScaler('cuda')

    def compute_vae_loss(self, x, recon_x, mu, logvar):
        """Compute VAE loss with KL divergence"""
        # Reconstruction loss (pixel-wise MSE)
        recon_loss = F.mse_loss(recon_x, x)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.beta * kl_loss

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # Forward pass
        with torch.amp.autocast('cuda'):
            mixed_z, mu , logvar = self.encoder(all_x)

            recon_x = self.decoder(mixed_z)

            if self.current_epoch % 100 == 0:
                save_images(all_x, recon_x, self.current_epoch)

            # Compute losses
            class_loss = F.cross_entropy(self.classifier(mixed_z), all_y)
            vae_loss = self.compute_vae_loss(all_x, recon_x, mu, logvar)

            # Combined loss với dynamic weighting
            total_loss = (
                class_loss
                + vae_loss
            )

        self.optimizer.zero_grad()
        self.grad_scaler.scale(total_loss).backward()

        # Gradient clipping
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        # Optimizer step with gradient scaling
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()
        self.current_epoch += 1

        return {
            "loss": total_loss.item(),
            "vae_loss": vae_loss.item(),
            "class_loss": class_loss.item()
        }

    def predict(self, x):
        mixed_z, _, _ = self.encoder(x)
        return self.classifier(mixed_z)


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(
            self.input_shape,
            self.num_classes,
            self.hparams,
            weights=self.network.state_dict(),
        ).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"],
        )
        self.network.reset_weights(meta_weights)

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, conditional, class_balance
    ):

        super(AbstractDANN, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.discriminator = networks.MLP(
            self.featurizer.n_outputs, num_domains, self.hparams
        )
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (
                list(self.discriminator.parameters())
                + list(self.class_embeddings.parameters())
            ),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def update(self, minibatches, unlabeled=None):
        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction="sum"),
            [disc_input],
            create_graph=True,
        )[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.0).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def update(self, minibatches, unlabeled=None):
        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]

        self.gan_transform = hparams["gan_transform"]

        device = next(self.network.parameters()).device
        # hparams['device'] = self.device
        if self.gan_transform:
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(
                self.hparams["mixup_alpha"], self.hparams["mixup_alpha"]
            )

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_meta_test = hparams["n_meta_test"]

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(
            minibatches, self.num_meta_test
        ):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(
                loss_inner_j, inner_net.parameters(), allow_unused=True
            )

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]

        self.gan_transform = hparams["gan_transform"]

        device = next(self.network.parameters()).device
        if self.gan_transform:
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=False
        )


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer(
            "embeddings", torch.zeros(num_domains, self.featurizer.n_outputs)
        )

        self.ema = self.hparams["mtl_ema"]

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = (
                self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]
            )

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(
                p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]

        self.gan_transform = hparams["gan_transform"]

        device = next(self.network_f.parameters()).device
        # hparams['device'] = self.device
        if self.gan_transform:
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p**2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "penalty": penalty.item()}


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {"loss": mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = mask.sum() / mask.numel()
            param.grad = mask * avg_grad
            param.grad *= 1.0 / (1e-10 + mask_t)

        return 0


class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(
                env_loss, self.network.parameters(), create_graph=True
            )

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(
            mean_loss, self.network.parameters(), retain_graph=True
        )

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams["penalty"] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": mean_loss.item(), "penalty": penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = (
            input_feat_size if input_feat_size == 2048 else input_feat_size * 2
        )

        self.cdpl = nn.Sequential(
            nn.Linear(input_feat_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_feat_size),
            nn.BatchNorm1d(input_feat_size),
        )

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]

        self.gan_transform = hparams["gan_transform"]

        device = next(self.network.parameters()).device
        # hparams['device'] = self.device
        self.cyclemixLayer = networks.CycleMix(hparams, device)

    def update(self, minibatches, unlabeled=None):

        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex == val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end - ex) + ex
            shuffle_indices2 = torch.randperm(end - ex) + ex
            for idx in range(end - ex):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.0)
        loss = cl_loss + C_scale * (
            lam * (L_ind_logit + L_ind_feat) + (1 - lam) * (L_hdl_logit + L_hdl_feat)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            betas=betas,
        )

        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(
                env_loss, self.network.parameters(), retain_graph=True
            )
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {"loss": mean_loss}

    def mask_grads(self, gradients, params):
        """
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        """
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(
                self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau)
            )
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = mask.sum() / mask.numel()
            param.grad = mask * avg_grad
            param.grad *= 1.0 / (1e-10 + mask_t)


class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert (
            backpack is not None
        ), "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]
        self.num_domains = num_domains
        self.gan_transform = hparams["gan_transform"]
        if self.gan_transform:
            device = next(self.network.parameters()).device
            self.cyclemixLayer = networks.CycleMix(hparams, device)

            self.num_domains = int(num_domains * 2)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams["nonlinear_classifier"],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction="none"))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

        # GAN AUGEMENTATION
        self.device = next(self.network.parameters()).device
        self.dataset = hparams["dataset"]

        self.gan_transform = hparams["gan_transform"]

        device = next(self.network.parameters()).device
        # hparams['device'] = self.device
        self.cyclemixLayer = networks.CycleMix(hparams, device)

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        if self.gan_transform:
            minibatches = self.cyclemixLayer(minibatches)

        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {
            "loss": objective.item(),
            "nll": all_nll.item(),
            "penalty": penalty.item(),
        }

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(
            dict_grads, len_minibatches
        )
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()),
                retain_graph=True,
                create_graph=True,
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx : all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (
                    (env_grads_centered).pow(2).mean(dim=0)
                )

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0,
                    ).mean(dim=0),
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)


class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [
            nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
            for i in range(num_domains + 1)
        ]
        self.olist = [
            torch.optim.SGD(
                self.clist[i].parameters(),
                lr=1e-1,
            )
            for i in range(num_domains + 1)
        ]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        # initial weights
        self.alpha = (
            torch.ones((num_domains, num_domains)).cuda()
            - torch.eye(num_domains).cuda()
        )

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(
                y, w, retain_graph=True, create_graph=True, allow_unused=True
            )
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.0
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams["iters"]:
            # TRM
            if self.hparams["class_balanced"]:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.0
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx : all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx : all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [
                    (
                        F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])
                        * (self.alpha[Q, i].data.detach())
                        if i in sample_list
                        else 0.0
                    )
                    for i in range(len(minibatches))
                ]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(
                    loss_P_sum, self.clist[Q].weight, create_graph=True
                )
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(
                    vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q])
                )

                loss_swap += loss_P_sum - self.hparams["cos_lambda"] * (
                    vec_grad_P.detach() @ vec_grad_Q
                )

                for i in sample_list:
                    self.alpha[Q, i] *= (
                        self.hparams["groupdro_eta"] * loss_P[i].data
                    ).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams["iters"]:
            loss_swap = loss + loss_swap
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {"nll": nll, "trm_loss": loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()


class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (
            self.hparams["ib_lambda"]
            if self.update_count >= self.hparams["ib_penalty_anneal_iters"]
            else 0.0
        )

        nll = 0.0
        ib_penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx : all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams["ib_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "IB_penalty": ib_penalty.item()}


class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.0).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        ib_penalty_weight = (
            self.hparams["ib_lambda"]
            if self.update_count >= self.hparams["ib_penalty_anneal_iters"]
            else 0.0
        )

        nll = 0.0
        irm_penalty = 0.0
        ib_penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx : all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if (
            self.update_count == self.hparams["irm_penalty_anneal_iters"]
            or self.update_count == self.hparams["ib_penalty_anneal_iters"]
        ):
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {
            "loss": loss.item(),
            "nll": nll.item(),
            "IRM_penalty": irm_penalty.item(),
            "IB_penalty": ib_penalty.item(),
        }


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, is_conditional):
        super(AbstractCAD, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = (
            is_conditional  # whether to use bottleneck conditioned on the label
        )
        self.base_temperature = 0.07
        self.temperature = hparams["temperature"]
        self.is_project = hparams["is_project"]  # whether apply projection head
        self.is_normalized = hparams[
            "is_normalized"
        ]  # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = (
            ~torch.eye(batch_size).bool().to(device)
        )  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (
            ~mask_d
        )  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = (
            mask_y.float(),
            mask_drop.float(),
            mask_y_n_d.float(),
            mask_y_d.float(),
        )

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = mask_y.sum(1) > 0
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = -(self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1
                )
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1
                )
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = mask_y_n_d.sum(1) > 0
            else:
                mask_valid = mask_y_d.sum(1) > 0

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if (
                self.is_flipped
            ):  # maximize log prob of samples from different domains and with same label
                bn_loss = -(self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1
                )
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1
                )

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams["lmbda"] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "clf_loss": clf_loss.item(),
            "bn_loss": bn_loss.item(),
            "total_loss": total_loss.item(),
        }

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z)
    - Require access to domain labels but not task labels
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(
            input_shape, num_classes, num_domains, hparams, is_conditional=False
        )


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(
            input_shape, num_classes, num_domains, hparams, is_conditional=True
        )


class Transfer(Algorithm):
    """Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)"""

    """ tries to ensure transferability among source domains, and thus transferabiilty between source and target"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))
        self.d_steps_per_g = hparams["d_steps_per_g"]

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams["gda"]:
            self.optimizer = torch.optim.SGD(
                self.adv_classifier.parameters(), lr=self.hparams["lr"]
            )
        else:
            self.optimizer = torch.optim.Adam(
                (
                    list(self.featurizer.parameters())
                    + list(self.classifier.parameters())
                ),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.adv_opt = torch.optim.SGD(
            self.adv_classifier.parameters(), lr=self.hparams["lr_d"]
        )

    def loss_gap(self, minibatches, device):
        """compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch"""
        max_env_loss, min_env_loss = torch.tensor(
            [-float("inf")], device=device
        ), torch.tensor([float("inf")], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams["t_lambda"] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams["t_lambda"] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(
                self.hparams["delta"], self.adv_classifier, self.classifier
            )
        return {"loss": loss.item(), "gap": -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams["t_lambda"] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {"loss": loss.item(), "gap": gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams["t_lambda"] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(
                self.hparams["delta"], self.adv_classifier, self.classifier
            )
            return {"gap": -gap.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    """Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class CausIRL_MMD(AbstractCausIRL):
    """Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


class CausIRL_CORAL(AbstractCausIRL):
    """Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=False
        )


class EQRM(ERM):
    """
    Empirical Quantile Risk Minimization (EQRM).
    Algorithm 1 from [https://arxiv.org/pdf/2207.09944.pdf].
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, dist=None):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))
        self.register_buffer(
            "alpha", torch.tensor(self.hparams["eqrm_quantile"], dtype=torch.float64)
        )
        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist

    def risk(self, x, y):
        return F.cross_entropy(self.network(x), y).reshape(1)

    def update(self, minibatches, unlabeled=None):
        env_risks = torch.cat([self.risk(x, y) for x, y in minibatches])

        if self.update_count < self.hparams["eqrm_burnin_iters"]:
            # Burn-in/annealing period uses ERM like penalty methods (which set penalty_weight=0, e.g. IRM, VREx.)
            loss = torch.mean(env_risks)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.alpha)

        if self.update_count == self.hparams["eqrm_burnin_iters"]:
            # Reset Adam (like IRM, VREx, etc.), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["eqrm_lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {"loss": loss.item()}
