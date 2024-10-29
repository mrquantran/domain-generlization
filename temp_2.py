class ALDModule(nn.Module):
    """Enhanced Automatic Latent Dimension module for domain generalization"""

    def __init__(
        self,
        initial_dim=100,
        min_dim=10,
        max_dim=200,
        reduction_rate=0.1,
        expansion_rate=0.1,
        eval_frequency=100,
        patience=5,
        metric_weights={"reconstruction": 0.4, "silhouette": 0.3, "fid": 0.3},
    ):
        super(ALDModule, self).__init__()
        self.current_dim = initial_dim
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.reduction_rate = reduction_rate
        self.expansion_rate = expansion_rate
        self.eval_frequency = eval_frequency
        self.patience = patience
        self.metric_weights = metric_weights

        # Metrics history with exponential moving average
        self.ema_alpha = 0.1
        self.metrics_history = {
            "reconstruction_loss": [],
            "silhouette_score": [],
            "fid_score": [],
            "domain_disc_score": [],  # New metric for domain discrimination
        }
        self.ema_metrics = {
            "reconstruction_loss": 0,
            "silhouette_score": 0,
            "fid_score": 0,
            "domain_disc_score": 0,
        }
        self.steps_no_improve = 0

    def update_ema_metrics(self, metrics):
        """Update exponential moving average of metrics"""
        for key, value in metrics.items():
            if key in self.ema_metrics:
                self.ema_metrics[key] = (
                    self.ema_alpha * value
                    + (1 - self.ema_alpha) * self.ema_metrics[key]
                )
                self.metrics_history[key].append(self.ema_metrics[key])

    def compute_clustering_metrics(self, latent_vectors, n_clusters=5):
        """Compute clustering quality metrics for latent space"""
        if len(latent_vectors) < n_clusters:
            return 0.0

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_vectors)

        # Compute silhouette score
        sil_score = silhouette_score(latent_vectors, cluster_labels)
        return sil_score

    def compute_fid_score(self, real_features, generated_features):
        """Compute Fréchet Inception Distance between real and generated features"""
        # Calculate mean and covariance for real and generated features
        mu1, sigma1 = real_features.mean(dim=0), torch.cov(real_features.T)
        mu2, sigma2 = generated_features.mean(dim=0), torch.cov(generated_features.T)

        # Calculate FID
        diff = mu1 - mu2
        covmean = torch.sqrt(sigma1 @ sigma2)

        if torch.is_complex(covmean):
            covmean = torch.real(covmean)

        fid = torch.sum(diff**2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    def compute_domain_discrimination(self, latent_vectors, domain_labels):
        """Compute domain discrimination score"""
        # Convert to numpy for wasserstein distance calculation
        latent_np = latent_vectors.detach().cpu().numpy()
        unique_domains = np.unique(domain_labels)

        # Calculate average Wasserstein distance between domain pairs
        distances = []
        for i in range(len(unique_domains)):
            for j in range(i + 1, len(unique_domains)):
                dom1_vectors = latent_np[domain_labels == unique_domains[i]]
                dom2_vectors = latent_np[domain_labels == unique_domains[j]]

                # Calculate distance for each dimension
                dim_distances = []
                for dim in range(dom1_vectors.shape[1]):
                    dist = wasserstein_distance(
                        dom1_vectors[:, dim], dom2_vectors[:, dim]
                    )
                    dim_distances.append(dist)
                distances.append(np.mean(dim_distances))

        return np.mean(distances) if distances else 0.0

    def should_adjust_dimension(self, metrics):
        """Determine if and how latent dimension should be adjusted"""
        # Update moving averages
        self.update_ema_metrics(metrics)

        # Compute weighted metric score
        weighted_score = (
            self.metric_weights["reconstruction"]
            * self.ema_metrics["reconstruction_loss"]
            + self.metric_weights["silhouette"] * self.ema_metrics["silhouette_score"]
            + self.metric_weights["fid"] * self.ema_metrics["fid_score"]
        )

        # Check recent history
        history_length = min(
            self.patience, len(self.metrics_history["reconstruction_loss"])
        )
        recent_scores = [
            (
                self.metric_weights["reconstruction"]
                * self.metrics_history["reconstruction_loss"][-i]
                + self.metric_weights["silhouette"]
                * self.metrics_history["silhouette_score"][-i]
                + self.metric_weights["fid"] * self.metrics_history["fid_score"][-i]
            )
            for i in range(1, history_length + 1)
        ]

        # Determine trend
        trend = np.mean(np.diff(recent_scores)) if len(recent_scores) > 1 else 0

        # Decision logic
        if trend > 0 and self.current_dim > self.min_dim:
            return "reduce"
        elif trend < 0 and self.current_dim < self.max_dim:
            return "expand"
        return "maintain"

    def adjust_dimension(self, action):
        """Adjust latent dimension based on action"""
        old_dim = self.current_dim

        if action == "reduce":
            reduction = max(int(self.current_dim * self.reduction_rate), 1)
            self.current_dim = max(self.current_dim - reduction, self.min_dim)
            print(f"Reducing dimension from {old_dim} to {self.current_dim}")
            return ("reduce", self.current_dim)

        elif action == "expand":
            expansion = max(int(self.current_dim * self.expansion_rate), 1)
            self.current_dim = min(self.current_dim + expansion, self.max_dim)
            print(f"Expanding dimension from {old_dim} to {self.current_dim}")
            return ("expand", self.current_dim)

        return ("maintain", self.current_dim)


def expand_layer(old_layer, layer_class, new_size):
    """Expands a layer by adding new neurons"""
    weights = old_layer.weight.data
    biases = old_layer.bias.data
    new_layer = layer_class(new_size[0], new_size[1])

    # Initialize new weights with small random values
    new_layer.weight.data.normal_(0, 0.02)
    new_layer.bias.data.zero_()

    # Copy old weights and biases
    new_layer.weight.data[: weights.shape[0], : weights.shape[1]] = weights
    new_layer.bias.data[: biases.shape[0]] = biases

    return new_layer


def downsize_layer(old_layer, layer_class, new_size, importance_scores=None):
    """Downsizes a layer by removing least important neurons"""
    weights = old_layer.weight.data
    biases = old_layer.bias.data

    if importance_scores is None:
        # Use L1 norm as importance if not provided
        importance_scores = torch.norm(weights, p=1, dim=1)

    # Get indices of most important neurons
    _, indices = torch.sort(importance_scores, descending=True)
    keep_indices = indices[: new_size[1]]

    # Create new layer
    new_layer = layer_class(new_size[0], new_size[1])
    new_layer.weight.data = weights[keep_indices, : new_size[0]]
    new_layer.bias.data = biases[keep_indices]

    return new_layer


class DimensionManager:
    """Manages dimension changes in the network"""

    def __init__(self, network):
        self.network = network

    def update_network_dimensions(self, action, new_dim, old_dim):
        device = next(self.network.parameters()).device

        """Updates network layers based on dimension change"""
        if action == "maintain":
            return

        # Get importance scores for neurons if reducing
        importance_scores = None
        if action == "reduce":
            importance_scores = self.compute_neuron_importance()

        # Update layers
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                old_size = (module.in_features, module.out_features)
                new_size = self.calculate_new_size(old_size, new_dim, old_dim)

                if action == "expand":
                    new_module = expand_layer(module, nn.Linear, new_size).to(device)
                else:
                    new_module = downsize_layer(
                        module, nn.Linear, new_size, importance_scores
                    ).to(device)

                # Replace old module
                parent = self.get_parent_module(name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, new_module)

    def compute_neuron_importance(self):
        """Compute importance scores for neurons"""
        importance_scores = {}

        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                # Use combination of L1 norm and gradient information
                weights = module.weight.data
                if module.weight.grad is not None:
                    grads = module.weight.grad.data
                    importance = torch.norm(weights, p=1, dim=1) * torch.norm(
                        grads, p=1, dim=1
                    )
                else:
                    importance = torch.norm(weights, p=1, dim=1)
                importance_scores[name] = importance

        return importance_scores

    def calculate_new_size(self, old_size, new_dim, old_dim):
        """Calculate new layer sizes based on dimension change"""
        in_scale = new_dim / old_dim
        out_scale = new_dim / old_dim

        new_in = max(int(old_size[0] * in_scale), 1)
        new_out = max(int(old_size[1] * out_scale), 1)

        return (new_in, new_out)

    def get_parent_module(self, name):
        """Get parent module for a given module name"""
        if "." not in name:
            return self.network

        parent_name = ".".join(name.split(".")[:-1])
        parent = self.network

        for part in parent_name.split("."):
            parent = getattr(parent, part)

        return parent


class GLOGenerator(nn.Module):
    """Generator network with proper dimension handling"""

    def __init__(self, latent_dim=100, output_shape=(3, 224, 224)):
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


# Modify GLOModule to use ALDModule
class GLOModule(nn.Module):
    def __init__(self, latent_dim=100, num_domains=3, batch_size=32, device="cuda"):
        super(GLOModule, self).__init__()
        self.device = device
        self.ald = ALDModule(
            initial_dim=latent_dim,
            metric_weights={"reconstruction": 0.4, "silhouette": 0.3, "fid": 0.3},
        ).to(self.device)

        # Add dimension validator
        self.input_validator = nn.Linear(self.ald.current_dim, self.ald.current_dim)

        # Initialize with proper output shape
        self.generator = GLOGenerator(
            latent_dim=self.ald.current_dim,
            output_shape=(3, 224, 224),
        ).to(self.device)

        # Add dimension adapter layer
        fc_output_size = (
            self.generator.init_h * self.generator.init_w * self.generator.init_channels
        )
        self.dim_adapter = nn.Sequential(
            nn.Linear(self.ald.current_dim, fc_output_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_output_size),
        ).to(device)

        self.domain_latents = nn.Parameter(
            torch.randn(
                num_domains, batch_size, self.ald.current_dim, device=self.device
            )
        )
        self._init_domain_latents()

        self.dim_manager = DimensionManager(self)

    def _init_domain_latents(self):
        nn.init.normal_(self.domain_latents, mean=0.0, std=0.02)

    def update_dimension(
        self, reconstruction_loss, silhouette_score, fid_score, domain_labels=None
    ):
        # Validate dimensions before update
        old_dim = self.ald.current_dim

        metrics = {
            "reconstruction_loss": reconstruction_loss,
            "silhouette_score": silhouette_score,
            "fid_score": fid_score,
            "domain_disc_score": (
                self.ald.compute_domain_discrimination(
                    self.domain_latents.view(-1, self.ald.current_dim), domain_labels
                )
                if domain_labels is not None
                else 0.0
            ),
        }

        action = self.ald.should_adjust_dimension(metrics)
        if action != "maintain":
            action, new_dim = self.ald.adjust_dimension(action)

            # Update generator if dimension changes
            if new_dim != old_dim:
                self.generator = GLOGenerator(
                    latent_dim=new_dim, output_shape=(3, 224, 224)
                ).to(self.device)

                # Update domain latents
                old_latents = self.domain_latents.data
                self.domain_latents = nn.Parameter(
                    torch.randn(old_latents.size(0), old_latents.size(1), new_dim),
                )
                self._init_domain_latents()

                # Update dimension manager
                self.dim_manager.update_network_dimensions(action, new_dim, old_dim)

    def forward(self, x, domain_idx):
        device = x.device
        batch_size = x.size(0)

        # Validate input dimension
        z = self.domain_latents[domain_idx, :batch_size].to(device)
        z = self.input_validator(z)

        # Adapt dimension for generator
        z = self.dim_adapter(z)
        z = z.view(
            batch_size,
            self.generator.init_channels,
            self.generator.init_h,
            self.generator.init_w,
        )

        # Generate output
        generated = self.generator.deconv(z)
        return generated, z.view(batch_size, -1)


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


class DimensionAdapter(nn.Module):
    """Adapter module to handle dimension mismatches"""

    def __init__(self, in_dim, out_dim):
        super(DimensionAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.adapter(x)


class CycleMixLayer(nn.Module):
    def __init__(self, hparams, device):
        super(CycleMixLayer, self).__init__()
        self.device = device
        self.sources = get_sources(hparams["dataset"], hparams["test_envs"])
        # Dynamic feature dimension based on backbone
        self.feature_dim = 512 if hparams.get("resnet18", True) else 2048

        # Add dimension adapter
        self.latent_adapter = DimensionAdapter(
            hparams.get("latent_dim", 100), self.feature_dim
        ).to(device)

        # GLO components
        self.glo = GLOModule(
            latent_dim=hparams.get("latent_dim", 100),
            num_domains=len(self.sources),
            batch_size=hparams["batch_size"],
            device=self.device,
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

        # Đảm bảo featurizer và input cùng device
        device = next(featurizer.parameters()).device
        x = x.to(device)
        y = y.to(device)

        # Generate domain-mixed samples
        x_hat, z = self.glo(x, domain_idx)

        # Extract and project features
        feat = featurizer(x)
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


class CYCLEMIX(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CYCLEMIX, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)

        device = next(self.network.parameters()).device
        self.cyclemixLayer = networks.CycleMixLayer(hparams, device)

        # Parameters
        self.reconstruction_lambda = hparams.get("reconstruction_lambda", 0.1)
        self.latent_reg_lambda = hparams.get("latent_reg_lambda", 0.01)
        self.contrastive_lambda = hparams.get("contrastive_lambda", 0.1)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters())
            + list(self.cyclemixLayer.projection.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.glo_optimizer = torch.optim.Adam(
            self.cyclemixLayer.glo.parameters(),
            lr=hparams.get("glo_lr", 1e-4),
            betas=(0.5, 0.999),
        )

    def compute_glo_loss(self, original, generated, latent):
        reconstruction_loss = F.mse_loss(generated, original)
        latent_reg = torch.mean(torch.norm(latent, dim=1))
        return (
            self.reconstruction_lambda * reconstruction_loss
            + self.latent_reg_lambda * latent_reg
        )

    def compute_contrastive_loss(self, projections):
        total_loss = 0
        for proj_tuple in projections:
            for i in range(len(proj_tuple)):
                for j in range(i + 1, len(proj_tuple)):
                    total_loss += self.cyclemixLayer.nt_xent(
                        proj_tuple[i], proj_tuple[j]
                    )
        return total_loss

    def update(self, minibatches, unlabeled=None):
        device = next(self.network.parameters()).device
        minibatches = [(x.to(device), y.to(device)) for x, y in minibatches]
        minibatches_aug, projections = self.cyclemixLayer(minibatches, self.featurizer)

        # Original and augmented samples
        orig_samples = minibatches_aug[: len(minibatches)]
        aug_samples = minibatches_aug[len(minibatches) :]

        # Classification
        all_x = torch.cat([x for x, y in orig_samples])
        all_y = torch.cat([y for x, y in orig_samples])
        class_loss = F.cross_entropy(self.predict(all_x), all_y)

        # GLO loss computation
        glo_loss = 0
        reconstruction_losses = []
        silhouette_scores = []
        fid_scores = []

        for (x_orig, _), (x_aug, _) in zip(orig_samples, aug_samples):
            _, z = self.cyclemixLayer.glo(x_orig, 0)
            current_glo_loss = self.compute_glo_loss(x_orig, x_aug, z)
            glo_loss += current_glo_loss

            # Compute metrics for ALD
            reconstruction_losses.append(F.mse_loss(x_aug, x_orig).item())

            # Compute Silhouette score on latent representations
            with torch.no_grad():
                z_np = z.cpu().numpy()
                if len(z_np) > 1:  # Need at least 2 samples
                    labels = np.random.randint(
                        0, 2, len(z_np)
                    )  # Generate random labels
                    silhouette_scores.append(silhouette_score(z_np, labels))

                # Compute FID score (simplified version)
                fid_scores.append(torch.norm(x_aug - x_orig).item())

        # Update latent dimension if needed
        avg_reconstruction = np.mean(reconstruction_losses)
        avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
        avg_fid = np.mean(fid_scores)

        self.cyclemixLayer.glo.update_dimension(
            avg_reconstruction, avg_silhouette, avg_fid
        )

        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(projections)

        # Total loss
        total_loss = class_loss + glo_loss + self.contrastive_lambda * contrastive_loss

        # Optimization
        self.optimizer.zero_grad()
        self.glo_optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.glo_optimizer.step()

        return {
            "loss": total_loss.item(),
            "class_loss": class_loss.item(),
            "glo_loss": glo_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "current_latent_dim": self.cyclemixLayer.glo.ald.current_dim,
        }

    def predict(self, x):
        return self.classifier(self.featurizer(x))
