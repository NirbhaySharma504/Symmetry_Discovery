import torch
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up project root and core directories robustly
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
CKPT = BASE_DIR / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)
FIGS = BASE_DIR / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# optimised data-loading params
NW, PM, PW, PF = 4, True, True, 2

def savefig_cached(fig, filename, dpi=180):
    """Save figure once; keep existing file as cache on reruns."""
    path = FIGS / filename
    if path.exists():
        print(f"  cached figure exists: {path}")
        return path
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  saved figure: {path}")
    return path

def show_cached_figure(filename, title=None, figsize=(10, 6)):
    """Display cached image and skip regeneration if it already exists."""
    path = FIGS / filename
    if not path.exists():
        return False
    print(f"  using cached figure: {path}")
    img = plt.imread(path)
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return True
