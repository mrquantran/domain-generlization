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
    def __init__(self, latent_dim, output_shape):
        super().__init__()

        self.output_shape = output_shape
        init_channels = 512

        self.fc = nn.Linear(latent_dim, init_channels * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(init_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_shape[0], 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, 4, 4)
        return self.decoder(x)

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
        self.domain_mu = nn.Parameter(torch.randn(num_domains, latent_dim))
        self.domain_logvar = nn.Parameter(torch.randn(num_domains, latent_dim))
        self.domain_latents = nn.Parameter(torch.randn(num_domains, batch_size, latent_dim))

        # Initialize with normal distribution as per GLO paper
        nn.init.normal_(self.domain_latents, mean=0.0, std=0.02)

    def forward(self, x, domain_idx):
        """Forward pass of the GLO module.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            domain_idx: Index of the current domain

        Returns:
            tuple: (generated images, latent vectors)
        """
        batch_size = x.size(0)

        # Encode input images
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # Create new domain latents tensor
        new_domain_latents = self.domain_latents.clone()
        new_domain_latents.data[domain_idx, :batch_size] = z.data
        self.domain_latents = nn.Parameter(new_domain_latents)

        # Generate images
        generated = self.generator(z)
        return generated, z

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Use EfficientNet as backbone
        self.backbone = torchvision.models.efficientnet_b1(
            weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT
        )
        backbone_out_features = self.backbone.classifier[1].in_features

        # Remove classifier
        self.backbone.classifier = nn.Identity()

        # VAE heads
        self.fc_mu = nn.Linear(backbone_out_features, latent_dim)
        self.fc_logvar = nn.Linear(backbone_out_features, latent_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_mu(features), self.fc_logvar(features)


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

    def process_domain(self, batch, domain_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process single domain data"""
        x, y = batch

        # Generate domain-mixed samples
        x_hat, z = self.glo(x, domain_idx)

        return (x, y), (x_hat, y), (z, y)

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

        # Unzip results
        original_samples, generated_samples, z_samples = zip(*processed_domains)

        # Return in required format
        return original_samples, generated_samples, z_samples

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
