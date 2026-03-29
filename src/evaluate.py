import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils import DEVICE, show_cached_figure, savefig_cached
from src.dataset import ROTATION_STEP, NUM_ROTATIONS

def visualise_supervised(vae, rot_mlp, latent_data, n_rows=4):
    """Apply the learned 30 deg transform iteratively and decode."""
    cache_name = "task2_supervised_iterative_rotation.png"
    if show_cached_figure(cache_name,
                          title="Task 2: Supervised Symmetry - Iterative 30 deg rotation",
                          figsize=(16, 4)):
        return

    z, y = latent_data["z"], latent_data["y"]
    unique_labels = sorted(y.unique().tolist())
    n_rows = min(n_rows, len(unique_labels))

    fig, axes = plt.subplots(n_rows, NUM_ROTATIONS + 1,
                             figsize=(2.2 * (NUM_ROTATIONS + 1), n_rows * 2.2))
    vae.eval()
    rot_mlp.eval()

    for row, lbl in enumerate(unique_labels[:n_rows]):
        # Start each trajectory from a canonical orientation (0 deg).
        mask = (y == lbl) & (latent_data["angle"] == 0)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        current = z[idx[0]].unsqueeze(0).to(DEVICE)

        for col in range(NUM_ROTATIONS + 1):
            with torch.no_grad():
                img = vae.decoder(current)[0, 0].cpu()
            axes[row, col].imshow(img, cmap="inferno")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{col * ROTATION_STEP} deg", fontsize=8)
            if col < NUM_ROTATIONS:
                with torch.no_grad():
                    current = rot_mlp(current)

    fig.suptitle("Task 2: Supervised Symmetry - Iterative 30 deg rotation")
    plt.tight_layout()
    savefig_cached(fig, cache_name)
    plt.show()

def visualise_unsupervised(vae, gen, latent_data, total_steps=25000,
                           n_show=10, n_rows=4, title="Unsupervised Symmetry",
                           fig_name="unsupervised_symmetry.png"):
    """Apply the discovered symmetry iteratively and decode.
    Uses many small epsilon-steps (25000), showing n_show snapshots."""
    if show_cached_figure(fig_name, title=title, figsize=(16, 5)):
        return

    z, y = latent_data["z"], latent_data["y"]
    unique_labels = sorted(y.unique().tolist())[:n_rows]
    # Uniformly sample snapshots along the full trajectory for visualization.
    show_every = max(1, total_steps // n_show)

    fig, axes = plt.subplots(len(unique_labels), n_show + 1,
                             figsize=(2 * (n_show + 1), 2 * len(unique_labels)))
    if len(unique_labels) == 1:
        axes = axes[np.newaxis, :]

    vae.eval()
    gen.eval()
    for row, lbl in enumerate(unique_labels):
        mask = (y == lbl) & (latent_data["angle"] == 0)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        current = z[idx[0]].unsqueeze(0).to(DEVICE)
        col = 0
        for s in range(total_steps + 1):
            if s % show_every == 0 and col <= n_show:
                with torch.no_grad():
                    img = vae.decoder(current)[0, 0].cpu()
                axes[row, col].imshow(img, cmap="inferno")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"step {s}", fontsize=6)
                col += 1
            if s < total_steps:
                with torch.no_grad():
                    current = gen(current)

    fig.suptitle(title)
    plt.tight_layout()
    savefig_cached(fig, fig_name)
    plt.show()

def plot_symmetry_paths(gen, latent_data, steps=25000,
                        record_every=100, title="Symmetry paths",
                        fig_name="symmetry_paths.png"):
    """Plot paths traced by the generator in latent space.
    Takes many tiny epsilon-steps and records every record_every-th point."""
    if show_cached_figure(fig_name, title=title, figsize=(8, 8)):
        return

    z, y = latent_data["z"], latent_data["y"]
    gen.eval()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z[:, 0], z[:, 1], c=y, s=0.5, alpha=0.2, cmap="RdBu")

    colors = plt.cm.Set1(np.linspace(0, 1, max(10, len(y.unique()))))
    for lbl in sorted(y.unique().tolist()):
        # Same 0 deg anchor so paths are comparable across labels.
        mask = (y == lbl) & (latent_data["angle"] == 0)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        current = z[idx[0]].unsqueeze(0).to(DEVICE)
        path = [current.cpu().numpy().squeeze()]
        for i in range(1, steps + 1):
            with torch.no_grad():
                current = gen(current)
            if i % record_every == 0:
                path.append(current.cpu().numpy().squeeze())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color=colors[lbl],
                linewidth=1.0, alpha=0.8)
        ax.plot(path[0, 0], path[0, 1], "o", color=colors[lbl], markersize=6)

    ax.set(title=title, xlabel="z1", ylabel="z2")
    plt.tight_layout()
    savefig_cached(fig, fig_name)
    plt.show()

def visualise_rotated_samples(train_data, test_data):
    """Visualise some rotated samples from train and test splits."""
    cache_name = "task1_rotated_samples.png"
    if show_cached_figure(cache_name, title="Rotated MNIST samples", figsize=(24, 4)):
        return

    fig, axes = plt.subplots(2, NUM_ROTATIONS, figsize=(24, 4))
    for i in range(NUM_ROTATIONS):
        for row, (ds, name) in enumerate([(train_data, "train"), (test_data, "test")]):
            mask = ds["angles"] == i * ROTATION_STEP
            axes[row, i].imshow(ds["images"][mask][0, 0].cpu(), cmap="gray")
            axes[row, i].set_title(f"{i * ROTATION_STEP} deg", fontsize=8)
            axes[row, i].axis("off")
    axes[0, 0].set_ylabel("train", fontsize=10)
    axes[1, 0].set_ylabel("test", fontsize=10)
    fig.suptitle("Task 1: Rotated MNIST samples (digits 0-1)")
    plt.tight_layout()
    savefig_cached(fig, cache_name)
    plt.show()

def visualise_latent_space(z_train, z_test):
    """Plot the latent space scatter for train and test."""
    cache_name = "task1_latent_space.png"
    if show_cached_figure(cache_name, title="Task 1: VAE Latent Space", figsize=(14, 6)):
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, z_data, title in [(axes[0], z_train, "Train"), (axes[1], z_test, "Test")]:
        z_arr = z_data["z"].cpu().numpy()
        y_arr = z_data["y"].cpu().numpy()
        sc = ax.scatter(z_arr[:, 0], z_arr[:, 1], c=y_arr, s=1, cmap="RdBu", alpha=0.5)
        fig.colorbar(sc, ax=ax)
        ax.set(title=f"Latent space - {title} (0-1)", xlabel="z1", ylabel="z2")
    fig.suptitle("Task 1: VAE Latent Space")
    plt.tight_layout()
    savefig_cached(fig, cache_name)
    plt.show()

def visualise_reconstructions(vae, test_data, num_samples=12):
    """Visualise VAE reconstructions."""
    cache_name = "task1_reconstructions.png"
    if show_cached_figure(cache_name, title="Task 1: VAE Reconstructions", figsize=(24, 4)):
        return

    vae.eval()
    sample_imgs = test_data["images"][:num_samples].to(DEVICE)
    with torch.no_grad():
        recon, _, _ = vae(sample_imgs)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    for i in range(num_samples):
        axes[0, i].imshow(sample_imgs[i, 0].cpu(), cmap="inferno")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="inferno")
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    fig.suptitle("Task 1: VAE Reconstructions")
    plt.tight_layout()
    savefig_cached(fig, cache_name)
    plt.show()

def visualise_rotation_trajectories(rot_mlp, latent_data):
    """Plot rotation trajectories in latent space."""
    cache_name = "task2_rotation_trajectories.png"
    if show_cached_figure(cache_name, title="Task 2: Rotation trajectories in latent space", figsize=(8, 8)):
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    z = latent_data["z"].cpu()
    y = latent_data["y"].cpu()
    ax.scatter(z[:, 0], z[:, 1], c=y, s=0.5, alpha=0.2, cmap="RdBu")

    rot_mlp.eval()
    unique_labels = sorted(y.unique().tolist())
    for lbl in unique_labels:
        # Trace a few example orbits per class for readability.
        mask = (y == lbl) & (latent_data["angle"].cpu() == 0)
        idx = mask.nonzero(as_tuple=True)[0][:3]  # 3 examples per digit
        for i in idx:
            current = z[i].unsqueeze(0).to(DEVICE)
            path = [current.cpu().numpy().squeeze()]
            for _ in range(NUM_ROTATIONS):
                with torch.no_grad():
                    current = rot_mlp(current)
                path.append(current.cpu().numpy().squeeze())
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], "o-", markersize=3, linewidth=1, alpha=0.7)

    ax.set(title="Task 2: Rotation trajectories in latent space", xlabel="z1", ylabel="z2")
    plt.tight_layout()
    savefig_cached(fig, cache_name)
    plt.show()

def visualise_reconstructions_full(vae_full, test_full, num_samples=12):
    """Visualise full VAE reconstructions (0-9)."""
    cache_name = "task3b_reconstructions_full.png"
    if show_cached_figure(cache_name, title="VAE (8D, low-β): original vs reconstruction (0-9)", figsize=(24, 4)):
        return

    vae_full.eval()
    sample = test_full["images"][:num_samples].to(DEVICE)
    with torch.no_grad():
        recon_f, _, _ = vae_full(sample)

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    for i in range(num_samples):
        axes[0, i].imshow(sample[i, 0].cpu(), cmap="inferno"); axes[0, i].axis("off")
        axes[1, i].imshow(recon_f[i, 0].cpu(), cmap="inferno"); axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Recon", fontsize=10)
    fig.suptitle("VAE (8D, low-β): original vs reconstruction (0-9)")
    plt.tight_layout()
    savefig_cached(fig, cache_name)
    plt.show()
