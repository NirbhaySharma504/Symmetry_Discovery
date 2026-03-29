import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Convolutional encoder with BatchNorm and LeakyReLU."""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),     # 28x28 -> 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),    # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),   # 7x7 -> 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    """Transposed-conv decoder with BatchNorm."""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 3, 2, 1),  # 4x4 -> 7x7
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 7x7 -> 14x14
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),     # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z))

class VAE(nn.Module):
    """Variational Autoencoder with reparameterisation trick."""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterise(mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        """Deterministic encoding (mean only)."""
        mu, _ = self.encoder(x)
        return mu

class RotationMLP(nn.Module):
    """Learn the latent-space transform for a 30 deg rotation."""
    def __init__(self, latent_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),     nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),     nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z):
        return self.net(z)

class LatentClassifier(nn.Module):
    """Classify digits from their latent representation."""
    def __init__(self, latent_dim=2, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, z):
        logits = self.net(z)
        # Keep binary output as shape [B] for BCEWithLogitsLoss convenience.
        return logits.squeeze(-1) if self.num_classes == 1 else logits

class SymmetryGenerator(nn.Module):
    """Discover symmetry transformations in latent space.
    Learns an infinitesimal generator W(z): z' = z + ε·W(z)
    preserves classifier output while moving along a symmetry orbit."""
    def __init__(self, latent_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, epsilon=1e-3):
        # Small epsilon turns the network output into an infinitesimal update.
        return z + epsilon * self.net(z)  # infinitesimal generator
