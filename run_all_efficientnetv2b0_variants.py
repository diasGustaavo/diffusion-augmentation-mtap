from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from efficientnetv2b0_kfold_runner import PipelineConfig, apply_smoke_test_defaults, run_pipeline

VARIANT_SPECS = [
    {"variant_name": "Variante_A", "methodology_label": "Variante A", "results_label": "Variante D", "dense_units": [512, 256], "dropout_rate": 0.2},
    {"variant_name": "Variante_B", "methodology_label": "Variante B", "results_label": "Variante C", "dense_units": [256, 128], "dropout_rate": 0.2},
    {"variant_name": "Variante_C", "methodology_label": "Variante C", "results_label": "Variante E", "dense_units": [512, 256, 128], "dropout_rate": 0.2},
    {"variant_name": "Variante_D", "methodology_label": "Variante D", "results_label": "Variante F", "dense_units": [128], "dropout_rate": 0.2},
    {"variant_name": "Variante_E", "methodology_label": "Variante E", "results_label": "", "dense_units": [96], "dropout_rate": 0.2},
    {"variant_name": "Variante_G", "methodology_label": "Variante F", "results_label": "Variante G", "dense_units": [64], "dropout_rate": 0.0},
    {"variant_name": "Variante_H", "methodology_label": "Variante F", "results_label": "Variante H", "dense_units": [64], "dropout_rate": 0.2},
]

VARIANT_SPECS_BY_NAME = {spec["variant_name"]: spec for spec in VARIANT_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EfficientNetV2B0 variants without relying on a notebook.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--augmentation-mode", choices=["none", "traditional"], default="none")
    parser.add_argument("--use-imagenet-weights", action="store_true")
    parser.add_argument("--unfreeze-base", action="store_true")
    parser.add_argument("--no-resume-training", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--optimizer-name", default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-min-lr", type=float, default=1e-7)
    parser.add_argument("--variant-name", action="append", default=[], help="Run only the given variants.")
    parser.add_argument("--isolate-variants", action="store_true", help="Run each variant in an isolated subprocess.")
    parser.add_argument("--python-bin", default="", help="Python used for isolated subprocesses.")
    return parser.parse_args()


def resolve_variant_specs(variant_names: list[str] | None) -> list[dict[str, object]]:
    if not variant_names:
        return [dict(spec) for spec in VARIANT_SPECS]

    resolved_specs: list[dict[str, object]] = []
    missing: list[str] = []
    for variant_name in variant_names:
        spec = VARIANT_SPECS_BY_NAME.get(variant_name)
        if spec is None:
            missing.append(variant_name)
            continue
        resolved_specs.append(dict(spec))

    if missing:
        available = ", ".join(sorted(VARIANT_SPECS_BY_NAME))
        requested = ", ".join(missing)
        raise ValueError(f"Variantes desconhecidas: {requested}. Disponiveis: {available}")

    return resolved_specs


def build_common_config(
    dataset_root: Path,
    augmentation_mode: str,
    use_imagenet_weights: bool,
    unfreeze_base: bool,
    resume_training: bool,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
    reduce_lr_patience: int,
    reduce_lr_min_lr: float,
) -> dict[str, object]:
    return {
        "dataset_root": str(dataset_root),
        "image_size": 224,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "n_splits": 5,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": 1e-4,
        "optimizer_name": optimizer_name,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": reduce_lr_patience,
        "reduce_lr_min_lr": reduce_lr_min_lr,
        "reduce_lr_min_delta": 1e-4,
        "augmentation_mode": augmentation_mode,
        "augmentation_rotation_deg": 5.0,
        "augmentation_width_shift": 0.1,
        "augmentation_height_shift": 0.1,
        "augmentation_shear": 0.1,
        "augmentation_zoom": 0.1,
        "augmentation_horizontal_flip": True,
        "augmentation_fill_mode": "nearest",
        "use_imagenet_weights": use_imagenet_weights,
        "freeze_base": not unfreeze_base,
        "resume_training": resume_training,
        "dataset_num_parallel_calls": -1,
        "dataset_prefetch_buffer": -1,
        "stop_on_min_lr_patience": 15,
        "resume_save_frequency_epochs": 25,
        "state_save_frequency_epochs": 5,
        "load_images_into_memory": True,
    }


def build_variant_config(
    variant: dict[str, object],
    output_dir: Path,
    common_config: dict[str, object],
    smoke_test: bool,
) -> PipelineConfig:
    aliases = [str(variant["methodology_label"])]
    if variant.get("results_label"):
        aliases.append(str(variant["results_label"]))

    config = PipelineConfig(
        output_dir=str(output_dir / str(variant["variant_name"]).lower()),
        dense_units=tuple(variant["dense_units"]),
        dropout_rate=float(variant["dropout_rate"]),
        variant_name=str(variant["variant_name"]),
        variant_aliases=tuple(aliases),
        **common_config,
    )
    if smoke_test:
        config = apply_smoke_test_defaults(config)
    return config


def summarize_artifacts(variant: dict[str, object], output_dir: Path, artifacts: dict[str, object]) -> dict[str, object]:
    return {
        "variant_name": variant["variant_name"],
        "methodology_label": variant["methodology_label"],
        "results_label": variant["results_label"],
        "output_dir": str(output_dir / str(variant["variant_name"]).lower()),
        "validation_accuracy": artifacts["validation_metrics"].get("accuracy"),
        "test_accuracy": artifacts["test_metrics"].get("accuracy"),
        "interrupted": artifacts.get("interrupted", False),
    }


def run_variants_inprocess(
    dataset_root: Path,
    output_dir: Path,
    augmentation_mode: str,
    use_imagenet_weights: bool,
    unfreeze_base: bool,
    resume_training: bool,
    smoke_test: bool,
    variant_specs: list[dict[str, object]],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
    reduce_lr_patience: int,
    reduce_lr_min_lr: float,
) -> list[dict[str, object]]:
    common_config = build_common_config(
        dataset_root=dataset_root,
        augmentation_mode=augmentation_mode,
        use_imagenet_weights=use_imagenet_weights,
        unfreeze_base=unfreeze_base,
        resume_training=resume_training,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        optimizer_name=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_min_lr=reduce_lr_min_lr,
    )

    all_artifacts: list[dict[str, object]] = []
    for variant in variant_specs:
        variant_name = str(variant["variant_name"])
        if is_variant_fully_cached(output_dir, variant_name):
            print(f"\n===== Variante {variant_name} ja concluida (cache). Reutilizando artifacts_summary.json. =====", flush=True)
            artifacts = load_variant_artifacts(output_dir, variant_name)
            all_artifacts.append(summarize_artifacts(variant, output_dir, artifacts))
            if artifacts.get("interrupted"):
                break
            continue
        config = build_variant_config(variant, output_dir, common_config, smoke_test=smoke_test)
        print(f"\n===== Running {variant_name} =====")
        artifacts = run_pipeline(config)
        all_artifacts.append(summarize_artifacts(variant, output_dir, artifacts))
        if artifacts.get("interrupted"):
            break
    return all_artifacts


def build_variant_subprocess_command(
    python_bin: str,
    dataset_root: Path,
    output_dir: Path,
    augmentation_mode: str,
    use_imagenet_weights: bool,
    unfreeze_base: bool,
    resume_training: bool,
    smoke_test: bool,
    variant_name: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
    reduce_lr_patience: int,
    reduce_lr_min_lr: float,
) -> list[str]:
    command = [
        python_bin,
        str(Path(__file__).resolve()),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(output_dir),
        "--augmentation-mode",
        augmentation_mode,
        "--variant-name",
        variant_name,
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--early-stopping-patience",
        str(early_stopping_patience),
        "--optimizer-name",
        optimizer_name,
        "--momentum",
        str(momentum),
        "--weight-decay",
        str(weight_decay),
        "--reduce-lr-patience",
        str(reduce_lr_patience),
        "--reduce-lr-min-lr",
        str(reduce_lr_min_lr),
    ]
    if use_imagenet_weights:
        command.append("--use-imagenet-weights")
    if unfreeze_base:
        command.append("--unfreeze-base")
    if not resume_training:
        command.append("--no-resume-training")
    if smoke_test:
        command.append("--smoke-test")
    return command


def load_variant_artifacts(output_dir: Path, variant_name: str) -> dict[str, object]:
    artifacts_path = output_dir / variant_name.lower() / "artifacts_summary.json"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Variant summary not found: {artifacts_path}")
    return json.loads(artifacts_path.read_text(encoding="utf-8"))


def is_variant_fully_cached(output_dir: Path, variant_name: str, expected_folds: int = 5) -> bool:
    variant_dir = output_dir / variant_name.lower()
    if not (variant_dir / "artifacts_summary.json").exists():
        return False
    for fold_index in range(1, expected_folds + 1):
        if not (variant_dir / f"fold_{fold_index}" / "best_model.keras").exists():
            return False
    return True


def run_variants_isolated(
    dataset_root: Path,
    output_dir: Path,
    augmentation_mode: str,
    use_imagenet_weights: bool,
    unfreeze_base: bool,
    resume_training: bool,
    smoke_test: bool,
    variant_specs: list[dict[str, object]],
    python_bin: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
    reduce_lr_patience: int,
    reduce_lr_min_lr: float,
) -> list[dict[str, object]]:
    all_artifacts: list[dict[str, object]] = []
    for variant in variant_specs:
        variant_name = str(variant["variant_name"])
        if is_variant_fully_cached(output_dir, variant_name):
            print(f"\n===== Variante {variant_name} ja concluida (cache). Reutilizando artifacts_summary.json. =====", flush=True)
            artifacts = load_variant_artifacts(output_dir, variant_name)
            all_artifacts.append(summarize_artifacts(variant, output_dir, artifacts))
            if artifacts.get("interrupted"):
                break
            continue
        print(f"\n===== Running {variant_name} in an isolated process =====")
        command = build_variant_subprocess_command(
            python_bin=python_bin,
            dataset_root=dataset_root,
            output_dir=output_dir,
            augmentation_mode=augmentation_mode,
            use_imagenet_weights=use_imagenet_weights,
            unfreeze_base=unfreeze_base,
            resume_training=resume_training,
            smoke_test=smoke_test,
            variant_name=variant_name,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            optimizer_name=optimizer_name,
            momentum=momentum,
            weight_decay=weight_decay,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_min_lr=reduce_lr_min_lr,
        )
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        return_code = subprocess.run(command, cwd=str(Path(__file__).resolve().parent), env=env, check=False).returncode
        if return_code != 0:
            raise RuntimeError(f"Failed to run {variant_name} in isolated subprocess. return_code={return_code}")
        artifacts = load_variant_artifacts(output_dir, variant_name)
        all_artifacts.append(summarize_artifacts(variant, output_dir, artifacts))
        if artifacts.get("interrupted"):
            break
    return all_artifacts


def run_all_variants(
    dataset_root: str | Path,
    output_dir: str | Path,
    augmentation_mode: str = "none",
    use_imagenet_weights: bool = False,
    unfreeze_base: bool = False,
    resume_training: bool = True,
    smoke_test: bool = False,
    variant_names: list[str] | None = None,
    isolate_processes: bool = False,
    python_bin: str = "",
    batch_size: int = 32,
    epochs: int = 300,
    learning_rate: float = 1e-5,
    early_stopping_patience: int = 30,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    reduce_lr_patience: int = 10,
    reduce_lr_min_lr: float = 1e-7,
) -> list[dict[str, object]]:
    resolved_dataset_root = Path(dataset_root).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_variant_specs = resolve_variant_specs(variant_names)
    resolved_python_bin = python_bin or sys.executable

    if isolate_processes and len(resolved_variant_specs) > 1:
        return run_variants_isolated(
            dataset_root=resolved_dataset_root,
            output_dir=resolved_output_dir,
            augmentation_mode=augmentation_mode,
            use_imagenet_weights=use_imagenet_weights,
            unfreeze_base=unfreeze_base,
            resume_training=resume_training,
            smoke_test=smoke_test,
            variant_specs=resolved_variant_specs,
            python_bin=resolved_python_bin,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            optimizer_name=optimizer_name,
            momentum=momentum,
            weight_decay=weight_decay,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_min_lr=reduce_lr_min_lr,
        )

    return run_variants_inprocess(
        dataset_root=resolved_dataset_root,
        output_dir=resolved_output_dir,
        augmentation_mode=augmentation_mode,
        use_imagenet_weights=use_imagenet_weights,
        unfreeze_base=unfreeze_base,
        resume_training=resume_training,
        smoke_test=smoke_test,
        variant_specs=resolved_variant_specs,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        optimizer_name=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_min_lr=reduce_lr_min_lr,
    )


def main() -> None:
    args = parse_args()
    all_artifacts = run_all_variants(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        augmentation_mode=args.augmentation_mode,
        use_imagenet_weights=args.use_imagenet_weights,
        unfreeze_base=args.unfreeze_base,
        resume_training=not args.no_resume_training,
        smoke_test=args.smoke_test,
        variant_names=args.variant_name,
        isolate_processes=args.isolate_variants,
        python_bin=args.python_bin,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        optimizer_name=args.optimizer_name,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_min_lr=args.reduce_lr_min_lr,
    )
    print("\nVariant summary:")
    print(json.dumps(all_artifacts, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
