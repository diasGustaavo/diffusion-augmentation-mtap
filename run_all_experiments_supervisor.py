from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervise run_all_experiments.py and relaunch automatically if the process dies."
    )
    parser.add_argument("--workspace-root", default="/home/guga/masters")
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--restart-delay-seconds", type=int, default=15)
    parser.add_argument("--max-restarts", type=int, default=0, help="0 significa ilimitado.")
    parser.add_argument("--child-log-path", default="")
    parser.add_argument("--supervisor-log-path", default="")
    parser.add_argument("--status-path", default="")
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
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--stall-timeout-seconds",
        type=int,
        default=900,
        help="Restart the child if the log has no updates for this period. 0 disables.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=10,
        help="Interval between progress checks of the child process.",
    )
    parser.add_argument(
        "--use-ionice",
        action="store_true",
        help="Run the child with ionice -c3. Disabled by default to avoid I/O starvation on WSL.",
    )
    return parser.parse_args()


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def write_status(status_path: Path, payload: dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_child_command(args: argparse.Namespace, workspace_root: Path, python_bin: str) -> list[str]:
    command = [python_bin, str(workspace_root / "run_all_experiments.py"), "--workspace-root", str(workspace_root)]
    if args.experiment_specs_path:
        command.extend(["--experiment-specs-path", args.experiment_specs_path])
    if args.output_suffix:
        command.extend(["--output-suffix", args.output_suffix])
    if args.summary_name:
        command.extend(["--summary-name", args.summary_name])
    if args.use_imagenet_weights:
        command.append("--use-imagenet-weights")
    if args.unfreeze_base:
        command.append("--unfreeze-base")
    command.extend(
        [
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--optimizer-name",
            args.optimizer_name,
            "--momentum",
            str(args.momentum),
            "--weight-decay",
            str(args.weight_decay),
            "--reduce-lr-patience",
            str(args.reduce_lr_patience),
            "--reduce-lr-min-lr",
            str(args.reduce_lr_min_lr),
        ]
    )
    if args.no_resume_training:
        command.append("--no-resume-training")
    if args.smoke_test:
        command.append("--smoke-test")
    return command


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    python_bin = args.python_bin or sys.executable
    child_log_path = Path(args.child_log_path or (workspace_root / "run_all_experiments_child.log"))
    supervisor_log_path = Path(args.supervisor_log_path or (workspace_root / "run_all_experiments_supervisor.log"))
    status_path = Path(args.status_path or (workspace_root / "outputs" / "run_all_experiments_supervisor_status.json"))

    restart_count = 0
    requested_stop = False
    child_process: subprocess.Popen[str] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal requested_stop, child_process
        requested_stop = True
        append_log(supervisor_log_path, f"Supervisor recebeu sinal {signum}. Encerrando.")
        if child_process is not None and child_process.poll() is None:
            try:
                child_process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    append_log(
        supervisor_log_path,
        f"Supervisor started. python_bin={python_bin} workspace_root={workspace_root} "
        f"resume={not args.no_resume_training} cpu_only={args.cpu_only} "
        f"stall_timeout={args.stall_timeout_seconds}s use_ionice={args.use_ionice}",
    )

    while not requested_stop:
        if args.max_restarts and restart_count > args.max_restarts:
            append_log(supervisor_log_path, f"Limite de reinicios atingido: {args.max_restarts}.")
            write_status(
                status_path,
                {
                    "status": "stopped_max_restarts",
                    "updated_at": now_iso(),
                    "restart_count": restart_count,
                    "child_log_path": str(child_log_path),
                    "supervisor_log_path": str(supervisor_log_path),
                },
            )
            break

        command = build_child_command(args, workspace_root, python_bin)
        if args.use_ionice:
            command = ["/usr/bin/ionice", "-c3", *command]
        append_log(supervisor_log_path, f"Starting child process: {' '.join(command)}")
        child_log_path.parent.mkdir(parents=True, exist_ok=True)

        with child_log_path.open("a", encoding="utf-8") as child_log:
            child_log.write(f"\n[{now_iso()}] ===== NOVA EXECUCAO DO FILHO =====\n")
            child_log.flush()
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            if args.cpu_only:
                env["CUDA_VISIBLE_DEVICES"] = "-1"
            else:
                env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            env.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
            child_process = subprocess.Popen(
                command,
                stdout=child_log,
                stderr=subprocess.STDOUT,
                cwd=str(workspace_root),
                text=True,
                env=env,
            )

            write_status(
                status_path,
                {
                    "status": "running",
                    "updated_at": now_iso(),
                    "restart_count": restart_count,
                    "pid": child_process.pid,
                    "child_log_path": str(child_log_path),
                    "supervisor_log_path": str(supervisor_log_path),
                    "command": command,
                },
            )

            last_log_mtime = child_log_path.stat().st_mtime if child_log_path.exists() else time.time()
            stalled = False
            return_code = None
            while True:
                polled_return_code = child_process.poll()
                if polled_return_code is not None:
                    return_code = polled_return_code
                    break

                time.sleep(max(1, args.poll_interval_seconds))

                try:
                    current_mtime = child_log_path.stat().st_mtime
                except FileNotFoundError:
                    current_mtime = last_log_mtime

                if current_mtime > last_log_mtime:
                    last_log_mtime = current_mtime
                    write_status(
                        status_path,
                        {
                            "status": "running",
                            "updated_at": now_iso(),
                            "restart_count": restart_count,
                            "pid": child_process.pid,
                            "child_log_path": str(child_log_path),
                            "supervisor_log_path": str(supervisor_log_path),
                            "command": command,
                            "last_log_update_at": datetime.fromtimestamp(current_mtime).astimezone().isoformat(timespec="seconds"),
                        },
                    )
                    continue

                if args.stall_timeout_seconds > 0 and (time.time() - last_log_mtime) >= args.stall_timeout_seconds:
                    stalled = True
                    append_log(
                        supervisor_log_path,
                        f"Stall detected: no new log lines for >= {args.stall_timeout_seconds}s. Restarting child.",
                    )
                    child_process.terminate()
                    try:
                        return_code = child_process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        append_log(supervisor_log_path, "Child did not exit after SIGTERM. Sending SIGKILL.")
                        child_process.kill()
                        return_code = child_process.wait(timeout=30)
                    break

        if requested_stop:
            write_status(
                status_path,
                {
                    "status": "stopped_by_signal",
                    "updated_at": now_iso(),
                    "restart_count": restart_count,
                    "return_code": return_code,
                    "child_log_path": str(child_log_path),
                    "supervisor_log_path": str(supervisor_log_path),
                },
            )
            break

        if return_code == 0:
            append_log(supervisor_log_path, "Child process finished successfully.")
            write_status(
                status_path,
                {
                    "status": "completed",
                    "updated_at": now_iso(),
                    "restart_count": restart_count,
                    "return_code": return_code,
                    "child_log_path": str(child_log_path),
                    "supervisor_log_path": str(supervisor_log_path),
                },
            )
            break

        restart_count += 1
        restart_reason = "stalled" if stalled else "crashed"
        append_log(
            supervisor_log_path,
            f"Child process {restart_reason} with return_code={return_code}. Restart {restart_count} in {args.restart_delay_seconds}s.",
        )
        write_status(
            status_path,
            {
                "status": "restarting",
                "updated_at": now_iso(),
                "restart_count": restart_count,
                "return_code": return_code,
                "restart_reason": restart_reason,
                "next_restart_in_seconds": args.restart_delay_seconds,
                "child_log_path": str(child_log_path),
                "supervisor_log_path": str(supervisor_log_path),
            },
        )
        time.sleep(args.restart_delay_seconds)


if __name__ == "__main__":
    main()
