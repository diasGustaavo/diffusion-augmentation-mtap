"""Visualize samples of real + synthetic images from the mix datasets."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

WORKSPACE = Path("/mnt/c/Users/ghmd1/Desktop/treinamento")
REPORTS = WORKSPACE / "reports"

CLASSES = [
    "167 - English foxhound",
    "265 - toy poodle",
    "585 - hair spray",
    "885 - velvet",
]


def grab(pattern_dir: Path, pattern_prefix: str, suffix: tuple[str, ...], limit: int):
    files = sorted([p for p in pattern_dir.iterdir() if p.name.startswith(pattern_prefix) and p.suffix in suffix])
    return files[:limit]


def sample_grid(dataset_name: str):
    ds_root = WORKSPACE / "datasets" / dataset_name / "treinamento"
    fig, axes = plt.subplots(len(CLASSES), 6, figsize=(16, 3.3 * len(CLASSES)))
    for row, cls in enumerate(CLASSES):
        cls_dir = ds_root / cls
        # 3 real (val_*.JPEG) + 3 synthetic (class_NNNNN_.png)
        real_files = sorted([p for p in cls_dir.iterdir() if p.name.startswith("val_")])[:3]
        synth_files = sorted([p for p in cls_dir.iterdir() if p.suffix == ".png"])[:3]
        entries = [(f, "REAL") for f in real_files] + [(f, "SYNTH") for f in synth_files]
        for col, (path, kind) in enumerate(entries):
            img = Image.open(path).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{kind}\n{cls.split(' - ')[1]}\n{path.name[:28]}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle(f"Amostras — {dataset_name} (3 reais + 3 sinteticas por classe)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPORTS / f"preview_{dataset_name}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print(f"Salvo: {out}")
    plt.close(fig)


if __name__ == "__main__":
    for ds in ("augmented_mix_5to1", "augmented_mix_2to1"):
        sample_grid(ds)
