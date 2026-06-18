from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an augmented dataset with a controlled mix of original and synthetic images."
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--original-root", default="datasets/original")
    parser.add_argument("--augmented-root", default="datasets/augmented")
    parser.add_argument("--output-root", default="datasets/augmented_controlled_mix")
    parser.add_argument("--synthetic-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(workspace_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def copy_split(src_root: Path, dst_root: Path) -> None:
    for src_path in sorted(src_root.rglob("*")):
        if src_path.is_dir():
            continue
        relative = src_path.relative_to(src_root)
        dst_path = dst_root / relative
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def build_controlled_mix_dataset(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    original_root = resolve_path(workspace_root, args.original_root)
    augmented_root = resolve_path(workspace_root, args.augmented_root)
    output_root = resolve_path(workspace_root, args.output_root)

    manifest_path = output_root / "build_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("validacao", "teste"):
        copy_split(original_root / split_name, output_root / split_name)

    train_original_root = original_root / "treinamento"
    train_augmented_root = augmented_root / "treinamento"
    class_dirs = sorted([item for item in train_original_root.iterdir() if item.is_dir()], key=lambda item: item.name)

    manifest: dict[str, object] = {
        "workspace_root": str(workspace_root),
        "original_root": str(original_root),
        "augmented_root": str(augmented_root),
        "output_root": str(output_root),
        "synthetic_per_class": args.synthetic_per_class,
        "seed": args.seed,
        "classes": [],
    }

    for class_index, class_dir in enumerate(class_dirs):
        rng = random.Random(args.seed + class_index)
        original_files = sorted([item for item in class_dir.iterdir() if item.is_file()], key=lambda item: item.name)
        augmented_class_root = train_augmented_root / class_dir.name
        augmented_files = sorted([item for item in augmented_class_root.iterdir() if item.is_file()], key=lambda item: item.name)
        original_hashes = {file_md5(path) for path in original_files}
        synthetic_candidates = [path for path in augmented_files if file_md5(path) not in original_hashes]
        if len(synthetic_candidates) < args.synthetic_per_class:
            raise ValueError(
                f"Class {class_dir.name} has only {len(synthetic_candidates)} synthetic images available, "
                f"mas foram solicitadas {args.synthetic_per_class}."
            )

        selected_synthetic = sorted(
            rng.sample(synthetic_candidates, k=args.synthetic_per_class),
            key=lambda item: item.name,
        )

        dst_class_root = output_root / "treinamento" / class_dir.name
        dst_class_root.mkdir(parents=True, exist_ok=True)

        for src_path in [*original_files, *selected_synthetic]:
            shutil.copy2(src_path, dst_class_root / src_path.name)

        manifest["classes"].append(
            {
                "class_name": class_dir.name,
                "original_count": len(original_files),
                "synthetic_count": len(selected_synthetic),
                "synthetic_files": [path.name for path in selected_synthetic],
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_controlled_mix_dataset(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
