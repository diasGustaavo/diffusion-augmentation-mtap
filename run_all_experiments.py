from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_all_efficientnetv2b0_variants import run_all_variants


DEFAULT_EXPERIMENT_SPECS = [
    {
        "name": "original",
        "dataset_root": "datasets/original",
        "output_dir_name": "original_efficientnetv2b0",
        "augmentation_mode": "none",
    },
    {
        "name": "augmented",
        "dataset_root": "datasets/augmented",
        "output_dir_name": "augmented_efficientnetv2b0",
        "augmentation_mode": "none",
    },
    {
        "name": "traditional_augmented",
        "dataset_root": "datasets/original",
        "output_dir_name": "original_efficientnetv2b0_traditional",
        "augmentation_mode": "traditional",
    },
    {
        "name": "generative_plus_traditional",
        "dataset_root": "datasets/augmented",
        "output_dir_name": "augmented_efficientnetv2b0_generative_plus_traditional",
        "augmentation_mode": "traditional",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4 EfficientNetV2B0 experiments in sequence, with automatic resume.")
    parser.add_argument("--workspace-root", default="/home/guga/masters")
    parser.add_argument("--experiment-specs-path", default="")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--summary-name", default="")
    parser.add_argument("--use-imagenet-weights", action="store_true")
    parser.add_argument("--unfreeze-base", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--optimizer-name", default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-min-lr", type=float, default=1e-7)
    parser.add_argument("--no-resume-training", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_workspace_path(workspace_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def load_experiment_specs(workspace_root: Path, experiment_specs_path: str) -> list[dict[str, object]]:
    if not experiment_specs_path:
        return [dict(spec) for spec in DEFAULT_EXPERIMENT_SPECS]

    specs_path = resolve_workspace_path(workspace_root, experiment_specs_path)
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The experiment file must contain a list of specifications.")

    specs: list[dict[str, object]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid specification at position {index}: expected JSON object.")
        specs.append(dict(item))
    return specs


def build_experiment_specs(
    workspace_root: Path,
    experiment_specs_path: str,
    output_suffix: str,
) -> list[dict[str, object]]:
    raw_specs = load_experiment_specs(workspace_root, experiment_specs_path)
    normalized_suffix = output_suffix.strip()
    if normalized_suffix:
        normalized_suffix = normalized_suffix.strip("_")

    resolved_specs: list[dict[str, object]] = []
    for spec in raw_specs:
        if "name" not in spec or "dataset_root" not in spec or "augmentation_mode" not in spec:
            raise ValueError(f"Incomplete specification: {spec}")

        output_dir_name = spec.get("output_dir_name")
        output_dir = spec.get("output_dir")
        if output_dir_name is None and output_dir is None:
            raise ValueError(f"Specification must provide output_dir_name or output_dir: {spec}")
        if output_dir_name is not None and output_dir is not None:
            raise ValueError(f"Specification must provide exactly one of output_dir_name and output_dir: {spec}")

        if output_dir_name is not None:
            output_dir_name = str(output_dir_name)
            if normalized_suffix:
                output_dir_name = f"{output_dir_name}_{normalized_suffix}"
            resolved_output_dir = (workspace_root / "outputs" / output_dir_name).resolve()
        else:
            resolved_output_dir = resolve_workspace_path(workspace_root, str(output_dir))

        resolved_specs.append(
            {
                "name": str(spec["name"]),
                "dataset_root": resolve_workspace_path(workspace_root, str(spec["dataset_root"])),
                "output_dir": resolved_output_dir,
                "augmentation_mode": str(spec["augmentation_mode"]),
            }
        )

    return resolved_specs


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    experiment_specs = build_experiment_specs(
        workspace_root=workspace_root,
        experiment_specs_path=args.experiment_specs_path,
        output_suffix=args.output_suffix,
    )

    all_experiments: list[dict[str, object]] = []
    for spec in experiment_specs:
        print(f"\n==============================")
        print(f"Running experiment: {spec['name']}")
        print(f"Dataset: {spec['dataset_root']}")
        print(f"Saida: {spec['output_dir']}")
        print(f"Aumento: {spec['augmentation_mode']}")
        print(f"Resume: {not args.no_resume_training}")
        print(f"==============================")

        variant_results = run_all_variants(
            dataset_root=spec["dataset_root"],
            output_dir=spec["output_dir"],
            augmentation_mode=spec["augmentation_mode"],
            use_imagenet_weights=args.use_imagenet_weights,
            unfreeze_base=args.unfreeze_base,
            resume_training=not args.no_resume_training,
            smoke_test=args.smoke_test,
            isolate_processes=True,
            python_bin=sys.executable,
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
        all_experiments.append(
            {
                "experiment_name": spec["name"],
                "dataset_root": str(spec["dataset_root"]),
                "output_dir": str(spec["output_dir"]),
                "augmentation_mode": spec["augmentation_mode"],
                "variant_results": variant_results,
            }
        )

    if args.summary_name:
        summary_name = args.summary_name
    elif args.output_suffix:
        summary_name = f"all_experiments_summary_{args.output_suffix.strip('_')}.json"
    else:
        summary_name = "all_experiments_summary.json"

    summary_path = workspace_root / "outputs" / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_experiments, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\nOverall summary:")
    print(json.dumps(all_experiments, indent=2, ensure_ascii=True))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
