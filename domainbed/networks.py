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
from domainbed.model.AlexNet import AlexNet

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class SSLRotationPredictor(nn.Module):
    def __init__(
        self,
        num_rotations=4,
        pretrained_model_path="/kaggle/input/rotnet/pytorch/default/1/model_net_epoch50",
    ):
        super(SSLRotationPredictor, self).__init__()
        # Load AlexNet model
        self.backbone = AlexNet({"num_classes": num_rotations})

        # Load pretrained weights if available
        if pretrained_model_path:
            self._load_pretrained_weights(pretrained_model_path)

        self.rotation_classifier = nn.Linear(
            4096, num_rotations
        )  # 4096 corresponds to the fc layer output

    def _load_pretrained_weights(self, path):
        try:
            checkpoint = torch.load(path)
            self.backbone.load_state_dict(checkpoint, strict=False)
            print("Pretrained RotNet weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def forward(self, x):
        # Forward pass through AlexNet backbone up to the fully connected layer
        features = self.backbone(
            x, out_feat_keys=["fc_block"]
        )  # Retrieve output from fc_block layer
        features = features.view(features.size(0), -1)  # Flatten for classifier input
        rotation_logits = self.rotation_classifier(features)
        return rotation_logits


class GLOGenerator(nn.Module):
    """Generator network following GLO paper architecture"""

    def __init__(self, latent_dim=100, output_dim=3):
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
    def __init__(self, latent_dim=100, num_domains=3, batch_size=32):
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
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),  # Thay thế ReLU
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


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
        self.feature_dim = 512 if hparams.get("resnet18", True) else 2048

        # GLO components (unchanged)
        self.glo = GLOModule(
            latent_dim=self.feature_dim,
            num_domains=len(self.sources),
            batch_size=hparams["batch_size"],
        ).to(device)

        # Cải tiến SSL Rotation Predictor
        self.ssl_rotation_predictor = SSLRotationPredictor(
            num_rotations=4  # Mở rộng góc xoay
        ).to(device)

        # Projection head với cấu trúc phức tạp hơn
        self.projection = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=hparams.get("proj_hidden_dim", 2048),
            output_dim=hparams.get("proj_output_dim", 256),
        ).to(device)

        # Contrastive loss
        self.nt_xent = NTXentLoss(temperature=hparams.get("temperature", 0.5))

        # SSL hyperparameters với adaptive lambda
        self.ssl_lambda = hparams.get("ssl_lambda", 0.1)
        self.rotation_choices = [0, 90, 180, 270]

        # Thêm mechanism để điều chỉnh SSL lambda động
        self.ssl_lambda_scheduler = self._create_ssl_lambda_scheduler()

    def _create_ssl_lambda_scheduler(self):
        def scheduler(epoch):
            # Phức tạp và linh hoạt hơn
            warmup_epochs = 30
            decay_start = 50
            min_lambda = 0.001

            # Cosine annealing with warm restart
            if epoch < warmup_epochs:
                return min(1.0, (epoch + 1) / warmup_epochs)

            T_0 = 50  # Restart period
            T_mult = 2  # Exponential restart

            def cosine_restart(epoch):
                return min_lambda + 0.5 * (1 - min_lambda) * (
                    1 + np.cos(np.pi * ((epoch - warmup_epochs) % T_0) / T_0)
                )

            return cosine_restart(epoch)

        return scheduler

    def rotate_image(self, x, rotation):
        """
        Advanced image rotation with robust handling

        Args:
            x (torch.Tensor): Input image tensor
            rotation (int): Rotation angle

        Returns:
            torch.Tensor: Rotated image tensor
        """
        import kornia
        import torch

        # Ensure tensor compatibility
        x = x.float() if not isinstance(x, torch.Tensor) else x

        # Handle different tensor dimensions
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Robust rotation using kornia
        try:
            angle = torch.tensor([rotation], dtype=torch.float32).to(x.device)
            center = torch.tensor(
                [x.shape[3] / 2, x.shape[2] / 2], dtype=torch.float32
            ).to(x.device)

            rotated_x = kornia.geometry.rotate(
                x,
                angle,
                center=center,
                mode="bilinear",
                padding_mode="reflection",  # More stable padding
            )

            return rotated_x
        except Exception as e:
            print(f"Rotation error: {e}")
            return x

    def process_domain_with_ssl(self, batch, domain_idx, featurizer, epoch):
        x, y = batch
        batch_size = x.size(0)
        # get image size
        size = x.size(-1)

        # Prepare augmented samples with rotations
        augmentations = []
        rotation_targets = []

        for i in range(batch_size):
            # Get individual image
            img = x[i].unsqueeze(0)

            # Apply random rotation from the 4 possible angles
            rotation = np.random.choice(self.rotation_choices)
            rotation_idx = self.rotation_choices.index(rotation)

            # Ensure image is correct size and format
            if img.size(-1) != size:
                img = F.interpolate(
                    img, size=(size, size), mode="bilinear", align_corners=False
                )

            # Apply rotation transformation
            rotated_img = self.rotate_image(img, rotation)
            augmentations.append(rotated_img)
            rotation_targets.append(rotation_idx)

        # Convert to batched tensor
        augmentations = torch.cat(augmentations, dim=0)
        rotation_targets = torch.tensor(rotation_targets, dtype=torch.long).to(
            self.device
        )

        # Generate GLO samples
        x_hat, z = self.glo(x, domain_idx)

        # Get rotation predictions using SSL rotation predictor
        with torch.no_grad():  # More efficient inference
            rotation_logits = self.ssl_rotation_predictor(augmentations)
            rotation_pred = F.softmax(rotation_logits, dim=1)

        # Calculate SSL rotation loss
        ssl_rotation_loss = F.cross_entropy(rotation_pred, rotation_targets)

        # Apply dynamic lambda scaling
        current_ssl_lambda = self.ssl_lambda_scheduler(epoch)
        scaled_ssl_loss = ssl_rotation_loss * current_ssl_lambda

        # Feature extraction and projection
        feat = featurizer(x)
        proj = self.projection(feat)
        proj = F.normalize(proj, dim=1)

        return (x, y), (x_hat, y), proj, scaled_ssl_loss

    def forward(self, x: list, featurizer, epoch):
        # Giữ nguyên logic của forward method
        processed_domains = [
            self.process_domain_with_ssl(batch, idx, featurizer, epoch)
            for idx, batch in enumerate(x)
        ]

        # Unzip results
        original_samples = [p[0] for p in processed_domains]
        generated_samples = [p[1] for p in processed_domains]
        projections = [p[2] for p in processed_domains]
        ssl_losses = [p[3] for p in processed_domains]

        all_samples = list(original_samples) + list(generated_samples)

        return all_samples, [projections], ssl_losses


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
