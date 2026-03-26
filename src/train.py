import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random

from src.models import VAE, RotationMLP, LatentClassifier, SymmetryGenerator
from src.utils import DEVICE, CKPT, savefig_cached, NW, PM, PW
from src.dataset import ROTATION_STEP

def vae_loss(recon, x, mu, logvar, beta=1.0):
    """ELBO loss = reconstruction (BCE) + beta * KL divergence."""
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld

def train_vae(train_loader, test_loader, latent_dim=2, epochs=100,
              lr=1e-3, beta_warmup=10, beta_max=1.0, name="vae"):
    """Train VAE with beta-warmup and ReduceLROnPlateau. Cached."""
    ckpt = CKPT / f"{name}.pth"
    if ckpt.exists():
        print(f"  loading VAE from {ckpt}")
        model = VAE(latent_dim).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        return model

    model = VAE(latent_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, verbose=True)

    hist = {"train": [], "val": []}
    print(f"  training VAE (latent_dim={latent_dim}, epochs={epochs}, β_max={beta_max})")

    for ep in range(1, epochs + 1):
        beta = min(beta_max, beta_max * ep / beta_warmup)

        model.train()
        t_loss = 0.0
        for imgs, _, _ in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_loss += loss.item()
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, _, _ in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                recon, mu, logvar = model(imgs)
                v_loss += vae_loss(recon, imgs, mu, logvar, beta).item()
        v_loss /= len(test_loader.dataset)

        sched.step(v_loss)
        hist["train"].append(t_loss)
        hist["val"].append(v_loss)

        if ep % 20 == 0 or ep == 1:
            print(f"    epoch {ep:>4d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}  β={beta:.4f}")

    torch.save(model.state_dict(), ckpt)
    print(f"  saved VAE to {ckpt}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["train"], label="train")
    ax.plot(hist["val"], label="val")
    ax.set(xlabel="epoch", ylabel="loss / sample", title="VAE Training Loss")
    ax.legend()
    plt.tight_layout()
    savefig_cached(fig, f"{name}_training_loss.png")
    plt.show()
    return model

def encode_dataset(model, data_loader, name):
    """Encode all images to latent space. Cached."""
    path = CKPT / f"{name}.pt"
    if path.exists():
        print(f"  loading latent codes from {path}")
        return torch.load(path, weights_only=False)
    model.eval()
    zs, ys, angles = [], [], []
    with torch.no_grad():
        for imgs, labels, angs in data_loader:
            z = model.encode(imgs.to(DEVICE, non_blocking=True))
            zs.append(z.cpu())
            ys.append(labels)
            angles.append(angs)
    result = {"z": torch.cat(zs), "y": torch.cat(ys), "angle": torch.cat(angles)}
    torch.save(result, path)
    print(f"  saved latent codes to {path}")
    return result

def build_rotation_pairs(latent_data):
    """Create (z, z_rotated_30) pairs using angle metadata."""
    z, y, a = latent_data["z"], latent_data["y"], latent_data["angle"]
    idx_map = {}
    for i in range(len(z)):
        idx_map.setdefault((int(y[i]), int(a[i])), []).append(i)
    src_list, tgt_list = [], []
    for (lbl, ang), indices in idx_map.items():
        next_ang = (ang + ROTATION_STEP) % 360
        if (lbl, next_ang) not in idx_map:
            continue
        tgt_indices = idx_map[(lbl, next_ang)]
        for si in indices:
            ti = random.choice(tgt_indices)
            src_list.append(z[si])
            tgt_list.append(z[ti])
    return torch.stack(src_list), torch.stack(tgt_list)

def train_rotation_mlp(latent_data, latent_dim=2, epochs=200,
                       lr=1e-3, batch_size=256, name="rot_mlp"):
    """Train the rotation MLP. Cached."""
    ckpt = CKPT / f"{name}.pth"
    pairs_cache = CKPT / f"pairs_{latent_dim}d.pt"

    if ckpt.exists():
        print(f"  loading RotationMLP from {ckpt}")
        model = RotationMLP(latent_dim).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        return model

    if pairs_cache.exists():
        print(f"  loading cached pairs from {pairs_cache}")
        src, tgt = torch.load(pairs_cache, weights_only=False)
    else:
        print("  building rotation pairs...")
        src, tgt = build_rotation_pairs(latent_data)
        torch.save((src, tgt), pairs_cache)

    ds = TensorDataset(src, tgt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=NW, pin_memory=PM, persistent_workers=PW)
    model = RotationMLP(latent_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  training RotationMLP (epochs={epochs})")
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for zs, zt in dl:
            zs = zs.to(DEVICE, non_blocking=True)
            zt = zt.to(DEVICE, non_blocking=True)
            loss = F.mse_loss(model(zs), zt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * zs.size(0)
        sched.step()
        if ep % 50 == 0 or ep == 1:
            print(f"    epoch {ep:>4d}/{epochs}  mse={ep_loss / len(ds):.6f}")

    torch.save(model.state_dict(), ckpt)
    print(f"  saved RotationMLP to {ckpt}")
    return model

def train_classifier(z_train, y_train, z_test, y_test,
                     latent_dim=2, num_classes=1, epochs=200,
                     lr=1e-3, name="clf"):
    """Train a classifier on latent vectors. Cached."""
    ckpt = CKPT / f"{name}.pth"
    if ckpt.exists():
        print(f"  loading classifier from {ckpt}")
        clf = LatentClassifier(latent_dim, num_classes).to(DEVICE)
        clf.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        return clf

    clf = LatentClassifier(latent_dim, num_classes).to(DEVICE)
    opt = optim.Adam(clf.parameters(), lr=lr)
    y_tr = y_train.float() if num_classes == 1 else y_train.long()
    ds = TensorDataset(z_train, y_tr)
    dl = DataLoader(ds, batch_size=256, shuffle=True,
                    num_workers=NW, pin_memory=PM, persistent_workers=PW)
    loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    print(f"  training classifier (classes={num_classes}, epochs={epochs})")
    for ep in range(1, epochs + 1):
        clf.train()
        total_loss = 0.0
        for zb, yb in dl:
            zb = zb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            loss = loss_fn(clf(zb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * zb.size(0)

        if ep % 50 == 0 or ep == 1:
            clf.eval()
            with torch.no_grad():
                zt = z_test.to(DEVICE)
                yt = y_test.to(DEVICE)
                if num_classes == 1:
                    preds = (torch.sigmoid(clf(zt)) > 0.5).float()
                    acc = (preds == yt.float()).float().mean()
                else:
                    preds = clf(zt).argmax(1)
                    acc = (preds == yt.long()).float().mean()
            print(f"    epoch {ep:>4d}/{epochs}  "
                  f"loss={total_loss / len(ds):.4f}  acc={acc:.4f}")

    torch.save(clf.state_dict(), ckpt)
    print(f"  saved classifier to {ckpt}")
    return clf

def train_symmetry_generator(classifier, z_data, latent_dim=2,
                             epochs=800, norm_weight=1.0,
                             eps=1e-4, lr=2e-3, name="gen"):
    """Train the symmetry generator using the paper's invariance + normalization loss.
    The generator learns an infinitesimal transform z' = z + ε·W(z)."""
    ckpt = CKPT / f"{name}.pth"
    if ckpt.exists():
        print(f"  loading generator from {ckpt}")
        gen = SymmetryGenerator(latent_dim).to(DEVICE)
        gen.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        return gen

    gen = SymmetryGenerator(latent_dim).to(DEVICE)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    opt = optim.Adam(gen.parameters(), lr=lr)

    # Use full dataset (no batching — matches reference for stability)
    z_all = z_data["z"].to(DEVICE)

    best_loss = float("inf")
    best_state = None

    print(f"  training SymmetryGenerator (epochs={epochs}, eps={eps})")
    for ep in range(1, epochs + 1):
        gen.train()
        opt.zero_grad()

        z_prime = gen(z_all, epsilon=eps)

        # 1) Invariance loss: oracle output preserved (scaled by ε²)
        logit_orig = classifier(z_all)
        logit_new = classifier(z_prime)
        inv_loss = ((logit_orig - logit_new) ** 2).mean() / (eps ** 2)

        # 2) Normalization loss: step size ≈ ε everywhere
        delta = z_prime - z_all
        step_norms = delta.norm(dim=1)
        norm_loss = (step_norms.mean() / eps - 1) ** 2 + step_norms.std() / eps

        loss = inv_loss + norm_weight * norm_loss
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in gen.state_dict().items()}

        if ep % 100 == 0 or ep == 1:
            print(f"    epoch {ep:>4d}/{epochs}  "
                  f"inv={inv_loss.item():.6f}  "
                  f"norm={norm_loss.item():.4f}  "
                  f"total={loss.item():.4f}")

    # Load best model
    if best_state is not None:
        gen.load_state_dict(best_state)

    torch.save(gen.state_dict(), ckpt)
    print(f"  saved generator to {ckpt}  (best_loss={best_loss:.4f})")
    return gen
