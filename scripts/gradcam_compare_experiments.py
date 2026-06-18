"""Compare Grad-CAM heatmaps across 4 experiments for the same image(s).

For a given rerun + variant + fold, loads the best_model.keras from:
  - original
  - augmented
  - traditional_augmented
  - generative_plus_traditional
Picks N test images (same across all experiments) and renders a grid
with original + 4 heatmaps per image.

Usage:
  python gradcam_compare_experiments.py [--rerun rerun1_imagenet_frozen]
                                        [--variant Variante_A]
                                        [--fold 1]
                                        [--images 3]
                                        [--seed 42]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


WORKSPACE = Path("/mnt/c/Users/ghmd1/Desktop/treinamento")
OUTPUTS = Path("/mnt/a/outputs")
REPORTS = WORKSPACE / "reports"
TEST_ROOT = WORKSPACE / "datasets" / "original" / "teste"

EXPERIMENTS = [
    ("original", "original_efficientnetv2b0"),
    ("augmented", "augmented_efficientnetv2b0"),
    ("traditional_augmented", "original_efficientnetv2b0_traditional"),
    ("generative_plus_traditional", "augmented_efficientnetv2b0_generative_plus_traditional"),
]

IMAGE_SIZE = (224, 224)


def _sorted_class_dirs() -> list[Path]:
    # Match runner ordering: string sort of folder names (not numeric).
    return sorted([p for p in TEST_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)


def class_names() -> list[str]:
    return [p.name.split(" - ", 1)[1] for p in _sorted_class_dirs()]


def build_test_records() -> list[tuple[int, str, Path]]:
    records = []
    for idx, folder in enumerate(_sorted_class_dirs()):
        label = folder.name.split(" - ", 1)[1]
        for img in sorted(folder.iterdir(), key=lambda p: p.name):
            records.append((idx, label, img))
    return records


def load_and_preprocess(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pil = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    display_array = np.array(pil, dtype=np.uint8)
    float_array = display_array.astype(np.float32)
    model_input = tf.keras.applications.efficientnet_v2.preprocess_input(float_array)
    return display_array, model_input


def make_gradcam(model: tf.keras.Model, image: np.ndarray) -> tuple[np.ndarray, int, float]:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("gradcam_target").output, model.output],
    )
    inputs = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]
    gradients = tape.gradient(class_channel, conv_outputs)
    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = float(tf.reduce_max(heatmap))
    if maximum == 0:
        heatmap_np = np.zeros(heatmap.shape, dtype=np.float32)
    else:
        heatmap_np = (heatmap / maximum).numpy()
    confidence = float(tf.nn.softmax(predictions[0])[class_index])
    return heatmap_np, class_index, confidence


def overlay(display_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (display_image.shape[1], display_image.shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    heatmap_array = np.array(heatmap_image, dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_array)[..., :3]
    merged = (display_image / 255.0) * (1 - alpha) + colored * alpha
    return np.clip(merged, 0, 1)


def model_path(rerun: str, output_dir_name: str, variant: str, fold: int) -> Path:
    variant_dir = variant.lower()
    return OUTPUTS / f"{output_dir_name}_{rerun}" / variant_dir / f"fold_{fold}" / "best_model.keras"


def pick_images(records: list[tuple[int, str, Path]], n: int, seed: int) -> list[tuple[int, str, Path]]:
    """Pick n images trying to cover each class at least once, then fill the rest randomly."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[int, str, Path]]] = {}
    for rec in records:
        by_class.setdefault(rec[0], []).append(rec)

    classes = list(by_class.keys())
    rng.shuffle(classes)
    chosen: list[tuple[int, str, Path]] = []
    used: set[str] = set()

    # one per class first
    for cls in classes:
        if len(chosen) >= n:
            break
        pick = rng.choice(by_class[cls])
        chosen.append(pick)
        used.add(str(pick[2]))

    # fill remaining from any class, avoiding duplicates
    if len(chosen) < n:
        all_remaining = [r for r in records if str(r[2]) not in used]
        rng.shuffle(all_remaining)
        for rec in all_remaining:
            if len(chosen) >= n:
                break
            chosen.append(rec)

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", default="rerun1_imagenet_frozen")
    parser.add_argument("--variant", default="Variante_A")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = class_names()
    records = build_test_records()
    chosen = pick_images(records, args.images, args.seed)

    # load models once
    loaded: list[tuple[str, tf.keras.Model]] = []
    for exp_name, output_dir_name in EXPERIMENTS:
        path = model_path(args.rerun, output_dir_name, args.variant, args.fold)
        if not path.exists():
            print(f"[WARN] skip {exp_name}: {path} not found", file=sys.stderr)
            continue
        print(f"[load] {exp_name}: {path}", flush=True)
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(path, compile=False)
        loaded.append((exp_name, model))

    n_images = len(chosen)
    n_cols = 1 + len(loaded)  # original + experiments
    fig, axes = plt.subplots(n_images, n_cols, figsize=(3.4 * n_cols, 3.6 * n_images), squeeze=False)
    for row, (true_label_idx, true_label, img_path) in enumerate(chosen):
        display_arr, model_input = load_and_preprocess(img_path)
        axes[row, 0].imshow(display_arr)
        axes[row, 0].set_title(f"Original\nReal: {true_label}", fontsize=10)
        axes[row, 0].axis("off")
        for col, (exp_name, model) in enumerate(loaded, start=1):
            with tf.device("/CPU:0"):
                heatmap, pred_idx, conf = make_gradcam(model, model_input)
            merged = overlay(display_arr, heatmap)
            axes[row, col].imshow(merged)
            pred_label = labels[pred_idx] if 0 <= pred_idx < len(labels) else "?"
            mark = "✓" if pred_idx == true_label_idx else "✗"
            axes[row, col].set_title(
                f"{exp_name}\n{mark} {pred_label} ({conf * 100:.1f}%)",
                fontsize=9,
            )
            axes[row, col].axis("off")

    fig.suptitle(
        f"Grad-CAM per experiment — rerun={args.rerun}, variant={args.variant}, fold={args.fold}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / f"gradcam_comparacao_experimentos_{args.rerun}_{args.variant.lower()}_fold{args.fold}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
