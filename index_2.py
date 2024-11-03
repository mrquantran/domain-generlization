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

        # Unknown domain: combine instance norm with domain mixing
        inst_norm = self.instance_norm(x.unsqueeze(1)).squeeze(1)
        gates = F.softmax(self.domain_gates, dim=0)

        out = 0
        for i in range(self.num_domains):
            domain_norm = self.norm[i](inst_norm)
            out += gates[i] * (self.scale[i] * domain_norm + self.bias[i])
        return out


class DomainRegularizer(nn.Module):
    """Feature regularization for domain-specific components"""

    def __init__(self, feature_dim, num_domains):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains

        # Domain consistency regularizer
        self.consistency_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
        )

        # Feature compactness regularizer
        self.compactness_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
        )

    def compute_consistency_loss(self, features, domain_idx):
        # Ensure consistent features within same domain
        projected = self.consistency_net(features)
        consistency_loss = F.mse_loss(projected, features.detach())
        return consistency_loss

    def compute_compactness_loss(self, features):
        # Encourage compact feature representations
        compactness_score = self.compactness_net(features)
        compactness_loss = torch.mean(torch.abs(compactness_score))
        return compactness_loss

    def forward(self, features, domain_idx=None):
        consistency_loss = self.compute_consistency_loss(features, domain_idx)
        compactness_loss = self.compute_compactness_loss(features)
        return consistency_loss + 0.1 * compactness_loss


class MemoryEfficientMixin:
    """Mixin class for memory-efficient training techniques"""

    def enable_gradient_checkpointing(self):
        # Enable gradient checkpointing for backbone
        self.backbone.gradient_checkpointing_enable()

    def get_mixed_precision_scaler(self):
        return torch.cuda.amp.GradScaler()

    def forward_chunk(self, x, chunk_size=32):
        """Forward pass in chunks to save memory"""
        if x.shape[0] <= chunk_size:
            return super().forward(x)

        outputs = []
        for chunk in x.split(chunk_size):
            with torch.cuda.amp.autocast():
                output = super().forward(chunk)
            outputs.append(output)

        # Combine chunks appropriately based on output type
        if isinstance(outputs[0], tuple):
            return tuple(
                torch.cat([o[i] for o in outputs]) for i in range(len(outputs[0]))
            )
        return torch.cat(outputs)

