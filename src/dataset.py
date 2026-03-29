import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from src.utils import CKPT, DATA_DIR, NW, PM, PW, PF

ROTATION_STEP = 30
NUM_ROTATIONS = 360 // ROTATION_STEP  # 12

def create_rotated(images, labels):
    """Rotate each image in 12 steps of 30 degrees."""
    all_i, all_l, all_a = [], [], []
    for ai in range(NUM_ROTATIONS):
        angle = ai * ROTATION_STEP
        # Keep labels aligned with each rotated copy, and track angle explicitly
        # so downstream code can form supervised rotation pairs.
        all_i.append(TF.rotate(images, angle))
        all_l.append(labels)
        all_a.append(torch.full((len(labels),), angle, dtype=torch.long))
    return torch.cat(all_i), torch.cat(all_l), torch.cat(all_a)

def load_data(max_digit=None, tag="rot"):
    """Load MNIST, optionally filter digits, rotate, and cache."""
    ct, ce = CKPT / f"{tag}_tr.pt", CKPT / f"{tag}_te.pt"
    if ct.exists() and ce.exists():
        print(f"  loading cached {tag}")
        return torch.load(ct, weights_only=False), torch.load(ce, weights_only=False)
    print(f"  building rotated dataset (max_digit={max_digit})")
    def go(split):
        ds = datasets.MNIST(DATA_DIR, train=(split == "train"), download=True,
                            transform=transforms.ToTensor())
        imgs = ds.data.unsqueeze(1).float() / 255.0
        labs = ds.targets
        if max_digit is not None:
            # Used in Task 1/3a to create the binary (0/1) subset.
            m = labs <= max_digit
            imgs, labs = imgs[m], labs[m]
        ri, rl, ra = create_rotated(imgs, labs)
        return {"images": ri, "labels": rl, "angles": ra}
    tr, te = go("train"), go("test")
    torch.save(tr, ct)
    torch.save(te, ce)
    return tr, te

def make_loader(d, bs=128, sh=True):
    # Centralized DataLoader settings keep train/eval behavior consistent.
    return DataLoader(
        TensorDataset(d["images"], d["labels"], d["angles"]),
        batch_size=bs, shuffle=sh, num_workers=NW,
        pin_memory=PM, persistent_workers=PW, prefetch_factor=PF)
