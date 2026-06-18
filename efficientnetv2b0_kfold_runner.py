from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PipelineConfig:
    dataset_root: str
    output_dir: str
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 300
    learning_rate: float = 1e-5
    n_splits: int = 5
    seed: int = 42
    dropout_rate: float = 0.2
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-4
    use_imagenet_weights: bool = True
    freeze_base: bool = True
    explainability_examples: int = 4
    max_train_samples_per_class: int | None = None
    max_eval_samples_per_class: int | None = None
    verbose: int = 1
    dense_units: tuple[int, ...] = (128,)
    optimizer_name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 10
    reduce_lr_min_lr: float = 1e-7
    reduce_lr_min_delta: float = 1e-4
    augmentation_mode: str = "none"
    augmentation_rotation_deg: float = 5.0
    augmentation_width_shift: float = 0.1
    augmentation_height_shift: float = 0.1
    augmentation_shear: float = 0.1
    augmentation_zoom: float = 0.1
    augmentation_horizontal_flip: bool = True
    augmentation_fill_mode: str = "nearest"
    variant_name: str = "default_variant"
    variant_aliases: tuple[str, ...] = ()
    resume_training: bool = True
    dataset_num_parallel_calls: int = -1
    dataset_prefetch_buffer: int = -1
    stop_on_min_lr_patience: int = 15
    resume_save_frequency_epochs: int = 10
    state_save_frequency_epochs: int = 5
    load_images_into_memory: bool = True


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow_runtime() -> None:
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[runtime] TensorFlow started with no visible GPU. Training will run on CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"[runtime] GPUs fisicas detectadas: {len(gpus)}")
    print(f"[runtime] GPUs logicas detectadas: {len(logical_gpus)}")
    for index, gpu in enumerate(gpus):
        print(f"[runtime] GPU {index}: {gpu.name}")


def clean_class_name(name: str) -> str:
    return re.sub(r"^\s*\d+\s*-\s*", "", name).strip()


def collect_split_records(
    split_dir: Path,
    seed: int,
    class_to_index: dict[str, int] | None = None,
    max_samples_per_class: int | None = None,
) -> pd.DataFrame:
    if not split_dir.exists():
        return pd.DataFrame(columns=["path", "label", "class_name", "display_name"])

    rng = np.random.default_rng(seed)
    class_dirs = sorted([item for item in split_dir.iterdir() if item.is_dir()], key=lambda item: item.name)

    if class_to_index is None:
        class_to_index = {directory.name: idx for idx, directory in enumerate(class_dirs)}

    rows: list[dict[str, Any]] = []
    for class_dir in class_dirs:
        if class_dir.name not in class_to_index:
            continue

        files = sorted(
            [file for file in class_dir.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda file: file.name,
        )
        if max_samples_per_class is not None and len(files) > max_samples_per_class:
            sampled = rng.choice(files, size=max_samples_per_class, replace=False)
            files = sorted(sampled, key=lambda file: file.name)

        for file in files:
            rows.append(
                {
                    "path": str(file),
                    "label": class_to_index[class_dir.name],
                    "class_name": class_dir.name,
                    "display_name": clean_class_name(class_dir.name),
                }
            )

    return pd.DataFrame(rows)


def make_image_data_generator(config: PipelineConfig, training: bool):
    if training and config.augmentation_mode == "traditional":
        return tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
            rotation_range=config.augmentation_rotation_deg,
            width_shift_range=config.augmentation_width_shift,
            height_shift_range=config.augmentation_height_shift,
            shear_range=config.augmentation_shear,
            zoom_range=config.augmentation_zoom,
            horizontal_flip=config.augmentation_horizontal_flip,
            fill_mode=config.augmentation_fill_mode,
        )

    return tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
    )


def build_image_data_generator(
    records: pd.DataFrame,
    config: PipelineConfig,
    training: bool,
    class_names: list[str],
):
    if records.empty:
        raise ValueError("No images found to build the generator.")

    dataframe = records.copy()
    dataframe["class_text"] = dataframe["display_name"]
    generator = make_image_data_generator(config=config, training=training)
    return generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col="path",
        y_col="class_text",
        classes=class_names,
        target_size=(config.image_size, config.image_size),
        color_mode="rgb",
        class_mode="sparse",
        batch_size=config.batch_size,
        shuffle=training,
        seed=config.seed,
    )


def load_images_to_numpy(records: pd.DataFrame, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    if records.empty:
        raise ValueError("No images found to load into memory.")

    # Keep images as uint8 to avoid blowing up RAM for larger folds.
    images = np.empty((len(records), image_size, image_size, 3), dtype=np.uint8)
    labels = records["label"].astype(np.int32).to_numpy()

    for index, image_path in enumerate(records["path"].astype(str).tolist()):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            images[index] = np.asarray(image, dtype=np.uint8)

    return images, labels


def derive_effective_verbose(config: PipelineConfig) -> int:
    # Keras progress bars explode log I/O when stdout is redirected.
    # In non-interactive runs, prefer one line per epoch.
    if config.verbose == 1 and not sys.stdout.isatty():
        return 2
    return config.verbose


def build_numpy_image_data_generator(
    images: np.ndarray,
    labels: np.ndarray,
    config: PipelineConfig,
    training: bool,
):
    generator = make_image_data_generator(config=config, training=training)
    return generator.flow(
        x=images,
        y=labels,
        batch_size=config.batch_size,
        shuffle=training,
        seed=config.seed,
    )


def build_tf_dataset_from_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    config: PipelineConfig,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(labels), seed=config.seed, reshuffle_each_iteration=True)

    def preprocess_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        return image, label

    parallel_calls = tf.data.AUTOTUNE if config.dataset_num_parallel_calls <= 0 else config.dataset_num_parallel_calls
    prefetch_buffer = tf.data.AUTOTUNE if config.dataset_prefetch_buffer <= 0 else config.dataset_prefetch_buffer

    dataset = dataset.map(preprocess_image, num_parallel_calls=parallel_calls)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset


def build_tf_dataset(records: pd.DataFrame, config: PipelineConfig, training: bool) -> tf.data.Dataset:
    if records.empty:
        raise ValueError("No images found to build the dataset.")

    paths = records["path"].astype(str).tolist()
    labels = records["label"].astype(np.int32).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(records), seed=config.seed, reshuffle_each_iteration=True)

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (config.image_size, config.image_size), antialias=True)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        return image, label

    parallel_calls = tf.data.AUTOTUNE if config.dataset_num_parallel_calls <= 0 else config.dataset_num_parallel_calls
    prefetch_buffer = tf.data.AUTOTUNE if config.dataset_prefetch_buffer <= 0 else config.dataset_prefetch_buffer

    dataset = dataset.map(load_image, num_parallel_calls=parallel_calls)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset


def build_model(config: PipelineConfig, num_classes: int) -> tf.keras.Model:
    weights = "imagenet" if config.use_imagenet_weights else None
    try:
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_shape=(config.image_size, config.image_size, 3),
        )
    except Exception as exc:
        print(f"[warning] Failed to load pretrained weights ({exc}). Continuing with random weights.")
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,
            input_shape=(config.image_size, config.image_size, 3),
        )

    base_model.trainable = not config.freeze_base

    inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3), name="image")
    x = inputs
    x = base_model(x, training=not config.freeze_base)
    x = tf.keras.layers.Activation("linear", name="gradcam_target")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    for layer_index, units in enumerate(config.dense_units, start=1):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{layer_index}")(x)
        x = tf.keras.layers.Dropout(config.dropout_rate, name=f"dropout_{layer_index}")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetv2b0_classifier")
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def release_memory(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    tf.keras.backend.clear_session()
    gc.collect()


class FoldStateCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        state_path: Path,
        history_path: Path,
        initial_history: dict[str, list[float]],
        initial_epoch: int,
        save_every_epochs: int,
    ) -> None:
        super().__init__()
        self.state_path = state_path
        self.history_path = history_path
        self.history = {key: [float(value) for value in values] for key, values in initial_history.items()}
        self.initial_epoch = initial_epoch
        self.save_every_epochs = max(1, save_every_epochs)

    def on_train_begin(self, logs=None) -> None:
        save_json(
            self.state_path,
            {
                "status": "running",
                "completed_epochs": self.initial_epoch,
            },
        )

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        if (epoch + 1) % self.save_every_epochs == 0:
            save_json(self.history_path, self.history)
            save_json(
                self.state_path,
                {
                    "status": "running",
                    "completed_epochs": epoch + 1,
                },
            )

    def mark_interrupted(self) -> None:
        completed_epochs = len(self.history.get("loss", []))
        save_json(self.history_path, self.history)
        save_json(
            self.state_path,
            {
                "status": "interrupted",
                "completed_epochs": completed_epochs,
            },
        )

    def mark_completed(self, summary: dict[str, Any]) -> None:
        completed_epochs = len(self.history.get("loss", []))
        save_json(self.history_path, self.history)
        save_json(
            self.state_path,
            {
                "status": "completed",
                "completed_epochs": completed_epochs,
                "summary": summary,
            },
        )


class GracefulInterruptHandler:
    def __init__(self) -> None:
        self.interrupted = False
        self._previous_handler = None

    def __enter__(self) -> "GracefulInterruptHandler":
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        signal.signal(signal.SIGINT, self._previous_handler)

    def _handle_signal(self, signum, frame) -> None:
        self.interrupted = True
        print("\n[warning] Interrupt received. Training will pause safely at the end of the current batch/epoch.")


class StopOnInterruptCallback(tf.keras.callbacks.Callback):
    def __init__(self, interrupt_handler: GracefulInterruptHandler) -> None:
        super().__init__()
        self.interrupt_handler = interrupt_handler

    def on_train_batch_end(self, batch: int, logs=None) -> None:
        if self.interrupt_handler.interrupted:
            self.model.stop_training = True


class StopOnMinLRStagnation(tf.keras.callbacks.Callback):
    def __init__(self, monitor: str, min_lr: float, min_delta: float, patience: int) -> None:
        super().__init__()
        self.monitor = monitor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.stagnant_epochs = 0

    def on_train_begin(self, logs=None) -> None:
        self.best = float("inf")
        self.stagnant_epochs = 0

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        current = float(current)
        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if current < self.best - self.min_delta:
            self.best = current
            self.stagnant_epochs = 0
            return

        if learning_rate <= self.min_lr + 1e-12:
            self.stagnant_epochs += 1
        else:
            self.stagnant_epochs = 0

        if self.stagnant_epochs >= self.patience:
            print(
                f"\n[early-stop] {self.monitor} stagnant for {self.stagnant_epochs} epochs "
                f"with minimum learning rate {learning_rate:.1e}."
            )
            self.model.stop_training = True


class PeriodicModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath: Path, save_every_epochs: int) -> None:
        super().__init__()
        self.filepath = filepath
        self.save_every_epochs = max(1, save_every_epochs)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if (epoch + 1) % self.save_every_epochs == 0:
            self.model.save(self.filepath, overwrite=True)

    def on_train_end(self, logs=None) -> None:
        self.model.save(self.filepath, overwrite=True)


def maybe_show_figure() -> None:
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


def plot_training_curves(histories: list[dict[str, list[float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold_index, history in enumerate(histories, start=1):
        epochs = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs, history["accuracy"], marker="o", label=f"Fold {fold_index} train")
        axes[0].plot(epochs, history["val_accuracy"], marker="x", linestyle="--", label=f"Fold {fold_index} validation")
        axes[1].plot(epochs, history["loss"], marker="o", label=f"Fold {fold_index} train")
        axes[1].plot(epochs, history["val_loss"], marker="x", linestyle="--", label=f"Fold {fold_index} validation")

    axes[0].set_title("Curvas de acuracia")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Acuracia")
    axes[1].set_title("Curvas de loss")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    maybe_show_figure()
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_title("Confusion matrix - test set")
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    maybe_show_figure()
    plt.close(fig)


def load_image_array(image_path: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    display_image = np.array(image)
    resized = image.resize((image_size, image_size))
    model_input = np.array(resized, dtype=np.float32)
    model_input = tf.keras.applications.efficientnet_v2.preprocess_input(model_input)
    return model_input, display_image


def make_gradcam_heatmap(model: tf.keras.Model, image_array: np.ndarray) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("gradcam_target").output, model.output],
    )

    inputs = tf.convert_to_tensor(image_array[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum) == 0.0:
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / maximum
    return heatmap.numpy()


def overlay_heatmap(display_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (display_image.shape[1], display_image.shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    heatmap_array = np.array(heatmap_image, dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_array)[..., :3]
    overlay = np.clip((display_image / 255.0) * (1 - alpha) + colored * alpha, 0, 1)
    return overlay


def choose_interesting_cases(
    records: pd.DataFrame,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    num_examples: int,
) -> pd.DataFrame:
    results = records.copy()
    results["predicted_label"] = predicted_labels
    results["confidence"] = probabilities.max(axis=1)
    results["correct"] = results["label"] == results["predicted_label"]

    chosen_indexes: list[int] = []

    confident_errors = results[~results["correct"]].sort_values("confidence", ascending=False).head(max(1, num_examples // 2))
    confident_correct = results[results["correct"]].sort_values("confidence", ascending=False).head(max(1, num_examples // 2))
    uncertain_cases = results.sort_values("confidence", ascending=True).head(num_examples)

    for frame in (confident_errors, confident_correct, uncertain_cases):
        for index in frame.index.tolist():
            if index not in chosen_indexes:
                chosen_indexes.append(index)
            if len(chosen_indexes) >= num_examples:
                break
        if len(chosen_indexes) >= num_examples:
            break

    return results.loc[chosen_indexes]


def plot_explainability_cases(
    records: pd.DataFrame,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    model_path: Path,
    config: PipelineConfig,
    output_path: Path,
) -> None:
    selected = choose_interesting_cases(records, probabilities, predicted_labels, config.explainability_examples)
    if selected.empty:
        print("[warning] No cases available for explainability.")
        return

    # Explainability is tiny compared to training; keep it on CPU to avoid
    # post-training CUDA/cuDNN instability during long experiment runs.
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(model_path)

        columns = 2
        rows = len(selected)
        fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
        if rows == 1:
            axes = np.array([axes])

        for row_index, (_, row) in enumerate(selected.iterrows()):
            image_array, display_image = load_image_array(Path(row["path"]), config.image_size)
            heatmap = make_gradcam_heatmap(model, image_array)
            overlay = overlay_heatmap(display_image, heatmap)

            true_label = class_names[int(row["label"])]
            predicted_label = class_names[int(row["predicted_label"])]
            confidence = float(row["confidence"])

            axes[row_index, 0].imshow(display_image)
            axes[row_index, 0].set_title(f"Original\nReal: {true_label}\nPredito: {predicted_label} ({confidence:.2%})")
            axes[row_index, 0].axis("off")

            axes[row_index, 1].imshow(overlay)
            axes[row_index, 1].set_title("Grad-CAM")
            axes[row_index, 1].axis("off")

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        maybe_show_figure()
        plt.close(fig)
    release_memory(model)


def evaluate_ensemble(
    model_paths: list[Path],
    records: pd.DataFrame,
    config: PipelineConfig,
    split_name: str,
    class_names: list[str],
    output_dir: Path,
    preloaded_images: np.ndarray | None = None,
    preloaded_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    if records.empty:
        return {"split": split_name, "available": False}

    print(
        f"[postprocess] Starting ensemble evaluation of split '{split_name}' with "
        f"{len(model_paths)} folds e {len(records)} amostras.",
        flush=True,
    )

    if preloaded_images is not None and preloaded_labels is not None:
        eval_images = preloaded_images
        eval_labels = preloaded_labels
        eval_inputs = tf.keras.applications.efficientnet_v2.preprocess_input(eval_images.astype(np.float32, copy=False))
        dataset = None
    elif config.load_images_into_memory:
        eval_images, eval_labels = load_images_to_numpy(records, config.image_size)
        eval_inputs = tf.keras.applications.efficientnet_v2.preprocess_input(eval_images.astype(np.float32, copy=False))
        dataset = None
    else:
        eval_images = None
        eval_labels = None
        eval_inputs = None
        dataset = build_tf_dataset(records, config=config, training=False)
    probabilities_per_fold: list[np.ndarray] = []

    for model_index, model_path in enumerate(model_paths, start=1):
        print(
            f"[postprocess] Split '{split_name}': inference of fold {model_index}/{len(model_paths)} "
            f"usando {Path(model_path).name}.",
            flush=True,
        )
        # Run inference on CPU in eager numpy batches to avoid XLA CPU JIT
        # compilation which can hang silently for hours inside model.predict().
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(model_path)
            if eval_inputs is not None:
                batches = [
                    model(eval_inputs[i : i + config.batch_size], training=False).numpy()
                    for i in range(0, len(eval_inputs), config.batch_size)
                ]
                probabilities = np.concatenate(batches, axis=0)
            else:
                batches = [model(batch[0], training=False).numpy() for batch in dataset]
                probabilities = np.concatenate(batches, axis=0)
        probabilities_per_fold.append(probabilities)
        release_memory(model)

    mean_probabilities = np.mean(probabilities_per_fold, axis=0)
    y_true = records["label"].astype(int).to_numpy()
    y_pred = mean_probabilities.argmax(axis=1)

    metrics = {
        "split": split_name,
        "available": True,
        "accuracy": float(np.mean(y_true == y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    if split_name == "test":
        plot_confusion(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            output_path=output_dir / "confusion_matrix_test.png",
        )
        print("[postprocess] Test confusion matrix generated.", flush=True)

    result = {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": mean_probabilities,
    }
    print(
        f"[postprocess] Ensemble evaluation of split '{split_name}' completed. "
        f"accuracy={metrics['accuracy']:.4f}",
        flush=True,
    )
    release_memory(dataset, eval_images, eval_labels, eval_inputs)
    return result


def print_dataset_summary(train_records: pd.DataFrame, validation_records: pd.DataFrame, test_records: pd.DataFrame) -> None:
    print("Dataset summary:")
    print(f"- total train: {len(train_records)}")
    print(f"- total external validation: {len(validation_records)}")
    print(f"- total test: {len(test_records)}")
    print("- images per class in training:")
    print(train_records["display_name"].value_counts().sort_index().to_string())


def summarize_variant(config: PipelineConfig) -> None:
    aliases = ", ".join(config.variant_aliases) if config.variant_aliases else "no aliases"
    dense_text = " -> ".join(str(unit) for unit in config.dense_units)
    print("\nExperiment configuration:")
    print(f"- variant: {config.variant_name}")
    print(f"- aliases: {aliases}")
    print(f"- dense layers: {dense_text}")
    print(f"- dropout: {config.dropout_rate}")
    print(f"- data augmentation: {config.augmentation_mode}")
    print(f"- batch size: {config.batch_size}")
    print(f"- max epochs: {config.epochs}")
    print(f"- learning rate: {config.learning_rate}")
    print(f"- momentum: {config.momentum}")
    print(f"- weight decay: {config.weight_decay}")
    print(f"- automatic resume: {config.resume_training}")


def build_fold_summary(history_dict: dict[str, list[float]], fold_index: int, fold_train: pd.DataFrame, fold_val: pd.DataFrame, checkpoint_path: Path) -> dict[str, Any]:
    best_epoch_index = int(np.argmax(history_dict["val_accuracy"]))
    return {
        "fold": fold_index,
        "train_samples": int(len(fold_train)),
        "validation_samples": int(len(fold_val)),
        "best_val_accuracy": float(history_dict["val_accuracy"][best_epoch_index]),
        "best_val_loss": float(history_dict["val_loss"][best_epoch_index]),
        "best_epoch": best_epoch_index + 1,
        "model_path": str(checkpoint_path),
    }


def build_interrupted_artifacts(
    config: PipelineConfig,
    dataset_root: Path,
    output_dir: Path,
    class_names: list[str],
    fold_summaries: list[dict[str, Any]],
    histories: list[dict[str, list[float]]],
    interrupted_fold: int,
) -> dict[str, Any]:
    if histories:
        plot_training_curves(histories, output_dir / "training_curves.png")

    artifacts = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "variant_name": config.variant_name,
        "variant_aliases": list(config.variant_aliases),
        "class_names": class_names,
        "fold_summaries": fold_summaries,
        "validation_metrics": {"available": False},
        "test_metrics": {"available": False},
        "best_model_path": None,
        "saved_files": {
            "curves": str(output_dir / "training_curves.png"),
            "confusion_matrix": str(output_dir / "confusion_matrix_test.png"),
            "explainability": str(output_dir / "gradcam_explainability_cases.png"),
            "fold_results": str(output_dir / "fold_results.json"),
        },
        "interrupted": True,
        "interrupted_fold": interrupted_fold,
        "resume_hint": "Run again with the same output_dir to continue from the last saved checkpoint.",
    }
    save_json(output_dir / "fold_results.json", fold_summaries)
    save_json(output_dir / "artifacts_summary.json", artifacts)
    return artifacts


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    tf.get_logger().setLevel("ERROR")
    configure_tensorflow_runtime()

    dataset_root = Path(config.dataset_root).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_root = dataset_root / "train"
    validation_root = dataset_root / "val"
    test_root = dataset_root / "test"

    class_directories = sorted([item for item in training_root.iterdir() if item.is_dir()], key=lambda item: item.name)
    class_to_index = {directory.name: idx for idx, directory in enumerate(class_directories)}
    class_names = [clean_class_name(directory.name) for directory in class_directories]

    train_records = collect_split_records(
        training_root,
        seed=config.seed,
        class_to_index=class_to_index,
        max_samples_per_class=config.max_train_samples_per_class,
    )
    validation_records = collect_split_records(
        validation_root,
        seed=config.seed,
        class_to_index=class_to_index,
        max_samples_per_class=config.max_eval_samples_per_class,
    )
    test_records = collect_split_records(
        test_root,
        seed=config.seed,
        class_to_index=class_to_index,
        max_samples_per_class=config.max_eval_samples_per_class,
    )
    effective_verbose = derive_effective_verbose(config)
    train_images_all = None
    train_labels_all = None
    validation_images_all = None
    validation_labels_all = None
    test_images_all = None
    test_labels_all = None

    if config.load_images_into_memory:
        print("[io] Preloading dataset into memory to avoid re-reading from disk between folds.")
        train_images_all, train_labels_all = load_images_to_numpy(train_records, config.image_size)
        validation_images_all, validation_labels_all = load_images_to_numpy(validation_records, config.image_size)
        test_images_all, test_labels_all = load_images_to_numpy(test_records, config.image_size)

    print_dataset_summary(train_records, validation_records, test_records)
    summarize_variant(config)

    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    histories: list[dict[str, list[float]]] = []
    fold_summaries: list[dict[str, Any]] = []
    model_paths: list[Path] = []

    for fold_index, (train_index, val_index) in enumerate(
        splitter.split(train_records["path"], train_records["label"]),
        start=1,
    ):
        print(f"\n========== Fold {fold_index}/{config.n_splits} ==========")
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "best_model.keras"
        latest_checkpoint_path = fold_dir / "latest_model.keras"
        history_path = fold_dir / "history.json"
        state_path = fold_dir / "fold_state.json"

        fold_train = train_records.iloc[train_index].reset_index(drop=True)
        fold_val = train_records.iloc[val_index].reset_index(drop=True)
        state = load_json(state_path, {})
        train_images = None
        train_labels = None
        val_images = None
        val_labels = None

        if (
            config.resume_training
            and state.get("status") == "completed"
            and checkpoint_path.exists()
            and history_path.exists()
            and "summary" in state
        ):
            print(f"[resume] Fold {fold_index} ja concluido. Reutilizando artefatos salvos.")
            fold_history = load_json(history_path, {})
            summary = dict(state["summary"])
            summary["model_path"] = str(checkpoint_path)
            histories.append(fold_history)
            fold_summaries.append(summary)
            model_paths.append(checkpoint_path)
            continue

        if config.augmentation_mode == "traditional":
            if config.load_images_into_memory:
                train_images = train_images_all[train_index]
                train_labels = train_labels_all[train_index]
                val_images = train_images_all[val_index]
                val_labels = train_labels_all[val_index]
                train_dataset = build_numpy_image_data_generator(
                    train_images,
                    train_labels,
                    config=config,
                    training=True,
                )
                val_dataset = build_numpy_image_data_generator(
                    val_images,
                    val_labels,
                    config=config,
                    training=False,
                )
            else:
                train_dataset = build_image_data_generator(
                    fold_train,
                    config=config,
                    training=True,
                    class_names=class_names,
                )
                val_dataset = build_image_data_generator(
                    fold_val,
                    config=config,
                    training=False,
                    class_names=class_names,
                )
        else:
            if config.load_images_into_memory:
                train_images = train_images_all[train_index]
                train_labels = train_labels_all[train_index]
                val_images = train_images_all[val_index]
                val_labels = train_labels_all[val_index]
                train_dataset = build_tf_dataset_from_arrays(train_images, train_labels, config=config, training=True)
                val_dataset = build_tf_dataset_from_arrays(val_images, val_labels, config=config, training=False)
            else:
                train_dataset = build_tf_dataset(fold_train, config=config, training=True)
                val_dataset = build_tf_dataset(fold_val, config=config, training=False)

        initial_history = load_json(history_path, {}) if config.resume_training else {}
        initial_epoch = 0

        if config.resume_training and state.get("status") in {"running", "interrupted"} and latest_checkpoint_path.exists():
            initial_epoch = int(state.get("completed_epochs", 0))
            print(f"[resume] Resuming fold {fold_index} from epoch {initial_epoch + 1}.")
            model = tf.keras.models.load_model(latest_checkpoint_path)
        else:
            model = build_model(config=config, num_classes=len(class_names))

        state_callback = None
        if config.resume_training:
            state_callback = FoldStateCallback(
                state_path=state_path,
                history_path=history_path,
                initial_history=initial_history,
                initial_epoch=initial_epoch,
                save_every_epochs=config.state_save_frequency_epochs,
            )
        interrupt_handler = GracefulInterruptHandler()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.reduce_lr_min_lr,
                min_delta=config.reduce_lr_min_delta,
                verbose=1,
            ),
            StopOnInterruptCallback(interrupt_handler),
            StopOnMinLRStagnation(
                monitor="val_loss",
                min_lr=config.reduce_lr_min_lr,
                min_delta=config.reduce_lr_min_delta,
                patience=config.stop_on_min_lr_patience,
            ),
        ]
        if state_callback is not None:
            callbacks.append(state_callback)
            callbacks.insert(
                2,
                PeriodicModelSaveCallback(
                    filepath=latest_checkpoint_path,
                    save_every_epochs=config.resume_save_frequency_epochs,
                ),
            )

        fit_history = None
        try:
            with interrupt_handler:
                fit_history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=config.epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    verbose=effective_verbose,
                )
        except KeyboardInterrupt:
            if state_callback is not None:
                state_callback.mark_interrupted()
            print(f"\n[interrupted] Training paused at fold {fold_index}.")
            release_memory(
                model,
                train_dataset,
                val_dataset,
                train_images,
                train_labels,
                val_images,
                val_labels,
            )
            return build_interrupted_artifacts(
                config=config,
                dataset_root=dataset_root,
                output_dir=output_dir,
                class_names=class_names,
                fold_summaries=fold_summaries,
                histories=histories,
                interrupted_fold=fold_index,
            )

        if interrupt_handler.interrupted:
            if state_callback is not None:
                state_callback.mark_interrupted()
                partial_history = state_callback.history
            else:
                partial_history = {
                    key: [float(value) for value in values]
                    for key, values in (fit_history.history if fit_history is not None else {}).items()
                }
            print(f"\n[interrupted] Training paused at fold {fold_index}.")
            release_memory(
                model,
                train_dataset,
                val_dataset,
                train_images,
                train_labels,
                val_images,
                val_labels,
            )
            return build_interrupted_artifacts(
                config=config,
                dataset_root=dataset_root,
                output_dir=output_dir,
                class_names=class_names,
                fold_summaries=fold_summaries,
                histories=histories + [partial_history],
                interrupted_fold=fold_index,
            )

        if state_callback is not None:
            history_dict = state_callback.history
        else:
            history_dict = {
                key: [float(value) for value in values]
                for key, values in (fit_history.history if fit_history is not None else {}).items()
            }
        save_json(history_path, history_dict)
        histories.append(history_dict)

        summary = build_fold_summary(
            history_dict=history_dict,
            fold_index=fold_index,
            fold_train=fold_train,
            fold_val=fold_val,
            checkpoint_path=checkpoint_path,
        )
        if state_callback is not None:
            state_callback.mark_completed(summary)
        fold_summaries.append(summary)
        model_paths.append(checkpoint_path)
        print(summary)
        release_memory(
            model,
            train_dataset,
            val_dataset,
            fold_train,
            fold_val,
            train_images,
            train_labels,
            val_images,
            val_labels,
        )

    print("[postprocess] Saving fold summary.", flush=True)
    save_json(output_dir / "fold_results.json", fold_summaries)
    print("[postprocess] Generating training curves.", flush=True)
    plot_training_curves(histories, output_dir / "training_curves.png")

    # Disable XLA/tf.function tracing for ensemble inference to avoid multi-hour
    # CPU JIT compilations that block post-processing silently.
    tf.config.run_functions_eagerly(True)

    print("[postprocess] Evaluating ensemble on external validation.", flush=True)
    validation_eval = evaluate_ensemble(
        model_paths=model_paths,
        records=validation_records,
        config=config,
        split_name="val",
        class_names=class_names,
        output_dir=output_dir,
        preloaded_images=validation_images_all,
        preloaded_labels=validation_labels_all,
    )
    print("[postprocess] Evaluating ensemble on the test set.", flush=True)
    test_eval = evaluate_ensemble(
        model_paths=model_paths,
        records=test_records,
        config=config,
        split_name="test",
        class_names=class_names,
        output_dir=output_dir,
        preloaded_images=test_images_all,
        preloaded_labels=test_labels_all,
    )

    best_fold = max(fold_summaries, key=lambda item: item["best_val_accuracy"])
    best_model_path = Path(best_fold["model_path"])

    if test_eval.get("metrics", {}).get("available", True):
        print("[postprocess] Generating Grad-CAM explainability cases.", flush=True)
        plot_explainability_cases(
            records=test_records,
            predicted_labels=test_eval["y_pred"],
            probabilities=test_eval["probabilities"],
            class_names=class_names,
            model_path=best_model_path,
            config=config,
            output_path=output_dir / "gradcam_explainability_cases.png",
        )
        print("[postprocess] Grad-CAM completed.", flush=True)

    artifacts = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "variant_name": config.variant_name,
        "variant_aliases": list(config.variant_aliases),
        "class_names": class_names,
        "fold_summaries": fold_summaries,
        "validation_metrics": validation_eval.get("metrics", {"available": False}),
        "test_metrics": test_eval.get("metrics", {"available": False}),
        "best_model_path": str(best_model_path),
        "saved_files": {
            "curves": str(output_dir / "training_curves.png"),
            "confusion_matrix": str(output_dir / "confusion_matrix_test.png"),
            "explainability": str(output_dir / "gradcam_explainability_cases.png"),
            "fold_results": str(output_dir / "fold_results.json"),
        },
        "interrupted": False,
    }
    print("[postprocess] Saving artifacts_summary.json.", flush=True)
    save_json(output_dir / "artifacts_summary.json", artifacts)

    print("\nFinal summary:")
    print(json.dumps(artifacts["validation_metrics"], indent=2, ensure_ascii=True))
    print(json.dumps(artifacts["test_metrics"], indent=2, ensure_ascii=True))

    release_memory(
        train_records,
        validation_records,
        test_records,
        train_images_all,
        train_labels_all,
        validation_images_all,
        validation_labels_all,
        test_images_all,
        test_labels_all,
    )

    return artifacts


def apply_smoke_test_defaults(config: PipelineConfig) -> PipelineConfig:
    return PipelineConfig(
        dataset_root=config.dataset_root,
        output_dir=config.output_dir,
        image_size=min(config.image_size, 96),
        batch_size=min(config.batch_size, 8),
        epochs=1,
        learning_rate=config.learning_rate,
        n_splits=config.n_splits,
        seed=config.seed,
        dropout_rate=config.dropout_rate,
        early_stopping_patience=1,
        early_stopping_min_delta=config.early_stopping_min_delta,
        use_imagenet_weights=False,
        freeze_base=True,
        explainability_examples=min(config.explainability_examples, 3),
        max_train_samples_per_class=config.max_train_samples_per_class or 8,
        max_eval_samples_per_class=config.max_eval_samples_per_class or 2,
        verbose=config.verbose,
        dense_units=config.dense_units,
        optimizer_name=config.optimizer_name,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        reduce_lr_factor=config.reduce_lr_factor,
        reduce_lr_patience=1,
        reduce_lr_min_lr=config.reduce_lr_min_lr,
        reduce_lr_min_delta=config.reduce_lr_min_delta,
        augmentation_mode=config.augmentation_mode,
        augmentation_rotation_deg=config.augmentation_rotation_deg,
        augmentation_width_shift=config.augmentation_width_shift,
        augmentation_height_shift=config.augmentation_height_shift,
        augmentation_shear=config.augmentation_shear,
        augmentation_zoom=config.augmentation_zoom,
        augmentation_horizontal_flip=config.augmentation_horizontal_flip,
        augmentation_fill_mode=config.augmentation_fill_mode,
        variant_name=config.variant_name,
        variant_aliases=config.variant_aliases,
        resume_training=config.resume_training,
        dataset_num_parallel_calls=-1,
        dataset_prefetch_buffer=-1,
        stop_on_min_lr_patience=min(config.stop_on_min_lr_patience, 2),
        resume_save_frequency_epochs=min(config.resume_save_frequency_epochs, 1),
        state_save_frequency_epochs=min(config.state_save_frequency_epochs, 1),
        load_images_into_memory=config.load_images_into_memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetV2B0 with 5-fold cross-validation on folder-structured datasets.")
    parser.add_argument("--dataset-root", required=True, help="Dataset base folder containing the training, validation and test splits.")
    parser.add_argument("--output-dir", default=None, help="Output folder for models and figures.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--max-train-samples-per-class", type=int, default=None)
    parser.add_argument("--max-eval-samples-per-class", type=int, default=None)
    parser.add_argument("--explainability-examples", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--dense-units", nargs="+", type=int, default=[128], help="List of neurons in the dense layers.")
    parser.add_argument("--optimizer-name", default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-min-lr", type=float, default=1e-7)
    parser.add_argument("--reduce-lr-min-delta", type=float, default=1e-4)
    parser.add_argument("--augmentation-mode", choices=["none", "traditional"], default="none")
    parser.add_argument("--augmentation-rotation-deg", type=float, default=5.0)
    parser.add_argument("--augmentation-width-shift", type=float, default=0.1)
    parser.add_argument("--augmentation-height-shift", type=float, default=0.1)
    parser.add_argument("--augmentation-shear", type=float, default=0.1)
    parser.add_argument("--augmentation-zoom", type=float, default=0.1)
    parser.add_argument("--augmentation-fill-mode", default="nearest")
    parser.add_argument("--no-augmentation-horizontal-flip", action="store_true")
    parser.add_argument("--variant-name", default="default_variant")
    parser.add_argument("--variant-alias", action="append", default=[])
    parser.add_argument("--no-resume-training", action="store_true")
    parser.add_argument("--dataset-num-parallel-calls", type=int, default=-1, help="-1 usa tf.data.AUTOTUNE.")
    parser.add_argument("--dataset-prefetch-buffer", type=int, default=-1, help="-1 usa tf.data.AUTOTUNE.")
    parser.add_argument("--stop-on-min-lr-patience", type=int, default=15)
    parser.add_argument("--resume-save-frequency-epochs", type=int, default=10)
    parser.add_argument("--state-save-frequency-epochs", type=int, default=5)
    parser.add_argument("--no-load-images-into-memory", action="store_true")
    parser.add_argument("--no-imagenet-weights", action="store_true")
    parser.add_argument("--unfreeze-base", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_root / "outputs_efficientnetv2b0"

    config = PipelineConfig(
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        n_splits=args.n_splits,
        seed=args.seed,
        dropout_rate=args.dropout_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_imagenet_weights=not args.no_imagenet_weights,
        freeze_base=not args.unfreeze_base,
        explainability_examples=args.explainability_examples,
        max_train_samples_per_class=args.max_train_samples_per_class,
        max_eval_samples_per_class=args.max_eval_samples_per_class,
        verbose=args.verbose,
        dense_units=tuple(args.dense_units),
        optimizer_name=args.optimizer_name,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_min_lr=args.reduce_lr_min_lr,
        reduce_lr_min_delta=args.reduce_lr_min_delta,
        augmentation_mode=args.augmentation_mode,
        augmentation_rotation_deg=args.augmentation_rotation_deg,
        augmentation_width_shift=args.augmentation_width_shift,
        augmentation_height_shift=args.augmentation_height_shift,
        augmentation_shear=args.augmentation_shear,
        augmentation_zoom=args.augmentation_zoom,
        augmentation_horizontal_flip=not args.no_augmentation_horizontal_flip,
        augmentation_fill_mode=args.augmentation_fill_mode,
        variant_name=args.variant_name,
        variant_aliases=tuple(args.variant_alias),
        resume_training=not args.no_resume_training,
        dataset_num_parallel_calls=args.dataset_num_parallel_calls,
        dataset_prefetch_buffer=args.dataset_prefetch_buffer,
        stop_on_min_lr_patience=args.stop_on_min_lr_patience,
        resume_save_frequency_epochs=args.resume_save_frequency_epochs,
        state_save_frequency_epochs=args.state_save_frequency_epochs,
        load_images_into_memory=not args.no_load_images_into_memory,
    )

    if args.smoke_test:
        config = apply_smoke_test_defaults(config)

    artifacts = run_pipeline(config)
    print("\nPipeline concluido.")
    print(json.dumps(artifacts["saved_files"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
