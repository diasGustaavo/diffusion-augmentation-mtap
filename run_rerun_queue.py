from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


QUEUE_ITEMS = [
    {
        "name": "rerun1_imagenet_frozen",
        "use_imagenet_weights": True,
        "unfreeze_base": False,
        "experiment_specs_path": "",
        "output_suffix": "rerun1_imagenet_frozen",
        "summary_name": "all_experiments_summary_rerun1_imagenet_frozen.json",
    },
    {
        "name": "rerun2_imagenet_unfrozen",
        "use_imagenet_weights": True,
        "unfreeze_base": True,
        "experiment_specs_path": "",
        "output_suffix": "rerun2_imagenet_unfrozen",
        "summary_name": "all_experiments_summary_rerun2_imagenet_unfrozen.json",
    },
    {
        "name": "rerun3_controlled_mix_imagenet_unfrozen",
        "use_imagenet_weights": True,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/controlled_mix.json",
        "output_suffix": "rerun3_controlled_mix_imagenet_unfrozen",
        "summary_name": "all_experiments_summary_rerun3_controlled_mix_imagenet_unfrozen.json",
    },
    {
        "name": "rerun4_from_scratch_unfrozen",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "",
        "output_suffix": "rerun4_from_scratch_unfrozen",
        "summary_name": "all_experiments_summary_rerun4_from_scratch_unfrozen.json",
    },
    {
        "name": "rerun5_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "",
        "output_suffix": "rerun5_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun5_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
    {
        "name": "rerun6_mix_5to1_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/mix_5to1.json",
        "output_suffix": "rerun6_mix_5to1_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun6_mix_5to1_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
    {
        "name": "rerun7_mix_2to1_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/mix_2to1.json",
        "output_suffix": "rerun7_mix_2to1_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun7_mix_2to1_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
    {
        "name": "rerun8_arquitetura_sem_generativa_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/arquitetura_sem_generativa.json",
        "output_suffix": "rerun8_arquitetura_sem_generativa_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun8_arquitetura_sem_generativa_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
    {
        "name": "rerun9_arquitetura_com_generativa_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/arquitetura_com_generativa.json",
        "output_suffix": "rerun9_arquitetura_com_generativa_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun9_arquitetura_com_generativa_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
    {
        "name": "rerun10_arquitetura_mix_5to1_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/arquitetura_com_generativa_mix_5to1.json",
        "output_suffix": "rerun10_arquitetura_mix_5to1_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun10_arquitetura_mix_5to1_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
    {
        "name": "rerun11_arquitetura_mix_2to1_from_scratch_unfrozen_adamw",
        "use_imagenet_weights": False,
        "unfreeze_base": True,
        "experiment_specs_path": "experiment_suites/arquitetura_com_generativa_mix_2to1.json",
        "output_suffix": "rerun11_arquitetura_mix_2to1_from_scratch_unfrozen_adamw",
        "summary_name": "all_experiments_summary_rerun11_arquitetura_mix_2to1_from_scratch_unfrozen_adamw.json",
        "batch_size": 32,
        "epochs": 300,
        "learning_rate": 3e-4,
        "early_stopping_patience": 50,
        "optimizer_name": "adamw",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "reduce_lr_patience": 15,
        "reduce_lr_min_lr": 1e-6,
        "stall_timeout_seconds": 3600,
    },
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa uma fila de reruns supervisionados e isolados.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def write_status(status_path: Path, payload: dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def get_summary_path(workspace_root: Path, item: dict[str, object]) -> Path:
    return workspace_root / "outputs" / str(item["summary_name"])


def is_item_completed(workspace_root: Path, item: dict[str, object]) -> bool:
    return get_summary_path(workspace_root, item).exists()


def build_supervisor_command(
    workspace_root: Path,
    python_bin: str,
    item: dict[str, object],
    cpu_only: bool,
) -> list[str]:
    logs_root = workspace_root / "rerun_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    command = [
        python_bin,
        str(workspace_root / "run_all_experiments_supervisor.py"),
        "--workspace-root",
        str(workspace_root),
        "--python-bin",
        python_bin,
        "--child-log-path",
        str(logs_root / f"{item['name']}_child_windows.log"),
        "--supervisor-log-path",
        str(logs_root / f"{item['name']}_supervisor_windows.log"),
        "--status-path",
        str(logs_root / f"{item['name']}_status.json"),
        "--output-suffix",
        str(item["output_suffix"]),
        "--summary-name",
        str(item["summary_name"]),
    ]
    if item.get("experiment_specs_path"):
        command.extend(["--experiment-specs-path", str(item["experiment_specs_path"])])
    if item.get("use_imagenet_weights"):
        command.append("--use-imagenet-weights")
    if item.get("unfreeze_base"):
        command.append("--unfreeze-base")
    if item.get("stall_timeout_seconds") is not None:
        command.extend(["--stall-timeout-seconds", str(item["stall_timeout_seconds"])])
    command.extend(
        [
            "--batch-size",
            str(item.get("batch_size", 32)),
            "--epochs",
            str(item.get("epochs", 300)),
            "--learning-rate",
            str(item.get("learning_rate", 1e-5)),
            "--early-stopping-patience",
            str(item.get("early_stopping_patience", 30)),
            "--optimizer-name",
            str(item.get("optimizer_name", "sgd")),
            "--momentum",
            str(item.get("momentum", 0.9)),
            "--weight-decay",
            str(item.get("weight_decay", 5e-4)),
            "--reduce-lr-patience",
            str(item.get("reduce_lr_patience", 10)),
            "--reduce-lr-min-lr",
            str(item.get("reduce_lr_min_lr", 1e-7)),
        ]
    )
    if cpu_only:
        command.append("--cpu-only")
    return command


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    python_bin = args.python_bin or sys.executable
    queue_status_path = workspace_root / "rerun_logs" / "rerun_queue_status.json"
    completed_items = [item["name"] for item in QUEUE_ITEMS if is_item_completed(workspace_root, item)]
    pending_items = [
        (index, item)
        for index, item in enumerate(QUEUE_ITEMS, start=1)
        if item["name"] not in completed_items
    ]

    if not pending_items:
        write_status(
            queue_status_path,
            {
                "status": "completed",
                "updated_at": now_iso(),
                "total_items": len(QUEUE_ITEMS),
                "completed_items": completed_items,
            },
        )
        return

    for index, item in pending_items:
        write_status(
            queue_status_path,
            {
                "status": "running",
                "updated_at": now_iso(),
                "current_item": item["name"],
                "current_index": index,
                "total_items": len(QUEUE_ITEMS),
                "completed_items": completed_items,
                "remaining_items": [pending_item["name"] for pending_index, pending_item in pending_items if pending_index >= index],
            },
        )
        command = build_supervisor_command(workspace_root, python_bin, item, args.cpu_only)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        return_code = subprocess.run(command, cwd=str(workspace_root), env=env, check=False).returncode
        if return_code != 0:
            write_status(
                queue_status_path,
                {
                    "status": "failed",
                    "updated_at": now_iso(),
                    "current_item": item["name"],
                    "current_index": index,
                    "total_items": len(QUEUE_ITEMS),
                    "return_code": return_code,
                    "completed_items": completed_items,
                },
            )
            raise SystemExit(return_code)

        completed_items.append(item["name"])

    write_status(
        queue_status_path,
        {
            "status": "completed",
            "updated_at": now_iso(),
            "total_items": len(QUEUE_ITEMS),
            "completed_items": completed_items,
        },
    )


if __name__ == "__main__":
    main()
