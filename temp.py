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
        minibatches_aug, projections = self.cyclemixLayer(minibatches, self.featurizer)

        # Separate original and augmented samples
        orig_samples = minibatches_aug[: len(minibatches)]
        aug_samples = minibatches_aug[len(minibatches) :]

        # Concatenate all samples for classification
        all_x = torch.cat([x for x, y in orig_samples])
        all_y = torch.cat([y for x, y in orig_samples])

        # Classification loss
        class_loss = F.cross_entropy(self.predict(all_x), all_y)

        # GLO loss computation
        glo_loss = 0
        for (x_orig, _), (x_aug, _) in zip(orig_samples, aug_samples):
            _, z = self.cyclemixLayer.glo(x_orig, 0)
            glo_loss += self.compute_glo_loss(x_orig, x_aug, z)

        # Contrastive loss computation
        contrastive_loss = self.compute_contrastive_loss(projections)

        # Total loss
        total_loss = class_loss + glo_loss + self.contrastive_lambda * contrastive_loss

        # Optimization steps
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
        }

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CycleMixLayer(nn.Module):
    def __init__(self, hparams, device):
        super(CycleMixLayer, self).__init__()
        self.device = device
        self.sources = get_sources(hparams["dataset"], hparams["test_envs"])
        # Dynamic feature dimension based on backbone
        self.feature_dim = 512 if hparams.get("resnet18", True) else 2048

        # GLO components
        self.glo = GLOModule(
            latent_dim=hparams.get("latent_dim", 100),
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
