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

        # Multi-head attention projections
        self.q_proj = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(self.n_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(self.n_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(self.n_heads)]
        )
        self.o_proj = nn.Linear(self.n_heads * latent_dim, latent_dim)

        # Layer norm for attention
        self.attention_norm = nn.LayerNorm(latent_dim)

        # Dynamic Neighborhood Aggregation
        self.domain_attention = nn.Linear(latent_dim, num_domains)

        # Hierarchical mixing
        self.hierarchical_level1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim),
                )
                for _ in range(num_domains)
            ]
        )

    def attention(self, mu, logvar, mask=None):
        """Multi-head attention with uncertainty weighting"""
        # Calculate uncertainty weights - shape: [batch_size, num_domains, latent_dim]
        uncertainty_weights = torch.exp(-logvar)
        uncertainty_weights = uncertainty_weights / uncertainty_weights.sum(
            dim=-1, keepdim=True
        )

        multi_head_outputs = []
        for head in range(self.n_heads):
            # Project inputs - shape: [batch_size, num_domains, latent_dim]
            query = self.q_proj[head](mu)
            key = self.k_proj[head](mu)
            value = self.v_proj[head](mu)

            # Weight projections by uncertainty
            query = query * uncertainty_weights
            key = key * uncertainty_weights

            # Compute attention scores - shape: [batch_size, num_domains, num_domains]
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            # Compute attention weights - shape: [batch_size, num_domains, num_domains]
            attention_weights = F.softmax(scores, dim=-1)

            # Compute mean uncertainty per domain - shape: [batch_size, num_domains, 1]
            domain_uncertainty = uncertainty_weights.mean(dim=-1, keepdim=True)

            # Apply uncertainty weighting to attention
            attention_weights = attention_weights * domain_uncertainty
            attention_weights = attention_weights / (
                attention_weights.sum(-1, keepdim=True) + 1e-6
            )

            # Apply attention to values
            head_output = torch.matmul(attention_weights, value)
            multi_head_outputs.append(head_output)

        # Combine heads - shape: [batch_size, num_domains, latent_dim]
        concat_output = torch.cat(multi_head_outputs, dim=-1)
        output = self.o_proj(concat_output)

        return output

    def dynamic_neighborhood_aggregation(self, features, logvars):
        # Calculate attention logits - shape: [batch_size, num_domains, num_domains]
        attn_logits = self.domain_attention(features)

        # Calculate mean uncertainty per domain - shape: [batch_size, num_domains, 1]
        uncertainty_weights = torch.exp(-logvars.mean(dim=-1, keepdim=True))

        # Apply temperature scaling and softmax
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)

        # Apply uncertainty weighting
        attn_weights = attn_weights * uncertainty_weights

        # Normalize weights
        attn_weights = torch.clamp(attn_weights, min=0.1, max=0.9)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        # Weighted aggregation - shape: [batch_size, num_domains, latent_dim]
        aggregated_features = torch.bmm(attn_weights, features)

        return aggregated_features

    def forward(self, x):
        # Split x into domains
        batch_size = x.shape[0] // self.num_domains  # 96 // 3 = 32
        x_domains = torch.split(
            x, batch_size
        )  # Split into list of [32, 3, 224, 224] tensors
        all_mus = []
        all_logvars = []

        for i, encoder in enumerate(self.domain_encoders):
            # Process each domain's data separately
            mu, logvar = encoder(x_domains[i])
            all_mus.append(mu)
            all_logvars.append(logvar)

        all_mus = torch.stack(all_mus, dim=1)  # [batch_size, num_domains, latent_dim]
        all_logvars = torch.stack(all_logvars, dim=1)

        # Apply attention with uncertainty
        attended_features = self.attention(all_mus, all_logvars)
        attended_features = self.attention_norm(
            attended_features + all_mus
        )  # skip connection

        # Dynamic neighborhood aggregation
        aggregated_features = self.dynamic_neighborhood_aggregation(
            attended_features, all_logvars
        )  # [batch_size, num_domains, latent_dim]

        # Level 1 mixing with uncertainty-aware aggregation
        level1_features = []
        for i in range(self.num_domains):
            # Extract features for current domain
            domain_attended = attended_features[:, i, :]  # [batch_size, latent_dim]
            domain_features = aggregated_features[:, i, :]  # [batch_size, latent_dim]

            # Calculate uncertainty weight
            uncertainty_weight = torch.exp(
                -all_logvars[:, i, :]
            )  # [batch_size, latent_dim]

            # Apply hierarchical mixing
            level1_mixed = self.hierarchical_level1[i](
                domain_features * uncertainty_weight
            )  # [batch_size, latent_dim]
            level1_mixed = level1_mixed + domain_attended  # Residual connection
            level1_features.append(level1_mixed)

        mixed_z = torch.cat(level1_features, dim=0)  # [batch_size * 3, latent_dim]
        mixed_mu = mixed_z.mean(dim=0)  # [latent_dim]
        mixed_logvar = mixed_z.var(dim=0).log()

        return mixed_z, mixed_mu, mixed_logvar


class GeneralizedEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(GeneralizedEncoder, self).__init__()

        print("Using ResNet50")
        self.network = torchvision.models.resnet50(pretrained=False)
        self.latent_dim = latent_dim

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
        self.network.fc = nn.Linear(
            2048, latent_dim
        )  # Change to output latent_dim dimension

        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size [batch_size, latent_dim]."""
        features = self.network(x)
        return features.view(
            features.size(0), -1
        )  # Reshape to [batch_size, latent_dim]

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


class CYCLEMIX(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CYCLEMIX, self).__init__(input_shape, num_classes, num_domains, hparams)

        # Giữ nguyên các hyperparameters hiện tại
        self.input_shape = input_shape
        self.latent_dim = hparams.get("latent_dim", 512)
        self.beta = hparams.get("beta", 4.0)
        self.grad_clip = hparams.get("grad_clip", 1.0)
        print(f"LATENT DIM: {self.latent_dim}")

        # Spatial dimensions
        self.h_dim = input_shape[1] // 16
        self.w_dim = input_shape[2] // 16

        # Encoder với hierarchical mixing
        self.encoder = MultiDomainVAEEncoder(
            input_shape=input_shape, latent_dim=self.latent_dim, num_domains=num_domains
        )

        self.generalized_encoder = GeneralizedEncoder(
            input_shape=input_shape, latent_dim=self.latent_dim
        )

        self.temperature = hparams.get("temperature", 2.0)

        # Giữ nguyên các components khác
        self.decoder = VAEDecoder(
            latent_dim=self.latent_dim,
            output_shape=input_shape,
            h_dim=self.h_dim,
            w_dim=self.w_dim,
        )

        self.domain_embeddings = nn.Parameter(torch.randn(num_domains, self.latent_dim))

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
            list(self.encoder.parameters())
            + list(self.classifier.parameters())
            + [self.domain_embeddings],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )

        self.generalized_encoder_optimizer = torch.optim.Adam(
            list(self.generalized_encoder.parameters()),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )

        # steps_per_epoch = hparams["steps_per_epoch"]
        # num_epochs = hparams["num_epochs"]
        # total_steps = int(steps_per_epoch * num_epochs)

        # self.total_steps = total_steps

        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=hparams["lr"] * 10,
        #     total_steps=self.total_steps,
        #     three_phase=False,
        # )

        self.current_epoch = 0
        self.grad_scaler = torch.amp.GradScaler("cuda")

    def compute_vae_loss(self, x, recon_x, mu, logvar):
        # Reconstruction loss (MSE for continuous data)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # Total loss with beta weighting for MMD
        total_loss = recon_loss

        return total_loss

    def compute_distillation_loss(self, student_features, teacher_features):
        """Knowledge distillation loss"""
        return F.kl_div(
            F.log_softmax(student_features / self.temperature, dim=-1),
            F.softmax(teacher_features / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature**2)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        class_loss = 0.0
        distill_loss = 0.0
        # Forward pass
        with torch.amp.autocast("cuda"):
            mixed_z, mu, logvar = self.encoder(all_x)

            # recon_x = self.decoder(mixed_z)

            # if self.current_epoch % 100 == 0:
            #     save_images(all_x, recon_x, self.current_epoch)

            gen_features = self.generalized_encoder(all_x)

            distill_loss = self.compute_distillation_loss(
                gen_features, mixed_z.detach()
            )

            # Compute losses
            class_loss = F.cross_entropy(self.classifier(mixed_z), all_y)
            # vae_loss = self.compute_vae_loss(all_x, recon_x, mu, logvar)

        # Train classifier
        self.optimizer.zero_grad()
        class_loss.backward()
        self.optimizer.step()

        # Train encoder
        self.generalized_encoder_optimizer.zero_grad()
        distill_loss.backward()
        self.generalized_encoder_optimizer.step()

        # self.scheduler.step()
        self.current_epoch += 1

        return {
            "distill_loss": distill_loss.item(),
            "class_loss": class_loss.item(),
        }

    def predict(self, x):
        mixed_z = self.generalized_encoder(x)
        return self.classifier(mixed_z)
