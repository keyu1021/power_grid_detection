from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence


SBATCH_JOB_ID_PATTERN = re.compile(r"\bSubmitted batch job (?P<job_id>\d+)\b")

RUNNING_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "PREEMPTED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "SUSPENDED",
    "STAGE_OUT",
}

SUCCESS_STATES = {"COMPLETED"}
FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "TIMEOUT",
}


class SlurmError(RuntimeError):
    """Raised when a Slurm command fails or returns an unexpected state."""


def _run_command(args: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        command = " ".join(shlex.quote(part) for part in args)
        raise SlurmError(
            f"Command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def submit_sbatch(
    script_path: str | Path,
    *,
    export_env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] | None = None,
    cwd: Path | None = None,
) -> str:
    args = ["sbatch"]
    if extra_args:
        args.extend(extra_args)
    if export_env:
        export_value = ",".join(["ALL", *[f"{key}={value}" for key, value in export_env.items()]])
        args.extend(["--export", export_value])
    args.append(str(script_path))

    completed = _run_command(args, cwd=cwd)
    match = SBATCH_JOB_ID_PATTERN.search(completed.stdout.strip())
    if match is None:
        raise SlurmError(f"Could not parse sbatch output:\n{completed.stdout}")
    return match.group("job_id")


def _normalize_state(state: str) -> str:
    token = state.strip().upper()
    if not token:
        return "UNKNOWN"
    token = token.split()[0]
    token = token.split("+", 1)[0]

    if token in SUCCESS_STATES:
        return "completed"
    if token in FAILED_STATES:
        if token == "TIMEOUT":
            return "timeout"
        if token == "CANCELLED":
            return "cancelled"
        return "failed"
    if token in RUNNING_STATES:
        return "running"
    return "unknown"


def _query_squeue(job_id: str, cwd: Path | None = None) -> str | None:
    completed = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return _normalize_state(lines[0])


def _query_sacct(job_id: str, cwd: Path | None = None) -> str | None:
    completed = subprocess.run(
        ["sacct", "-n", "-P", "-j", job_id, "--format=State"],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None

    states: list[str] = []
    for raw_line in completed.stdout.splitlines():
        token = raw_line.strip().split("|", 1)[0].strip()
        if token:
            states.append(token)

    for state in states:
        normalized = _normalize_state(state)
        if normalized != "unknown":
            return normalized

    return None


def get_job_state(job_id: str, *, cwd: Path | None = None) -> str:
    state = _query_squeue(job_id, cwd=cwd)
    if state is not None:
        return state

    state = _query_sacct(job_id, cwd=cwd)
    if state is not None:
        return state

    return "missing"


def wait_for_job(
    job_id: str,
    *,
    cwd: Path | None = None,
    poke_interval: int = 30,
    timeout: int = 6 * 60 * 60,
    missing_confirmations: int = 3,
) -> str:
    deadline = time.monotonic() + timeout
    missing_count = 0

    while True:
        if time.monotonic() > deadline:
            raise SlurmError(f"Timed out waiting for Slurm job {job_id}")

        state = get_job_state(job_id, cwd=cwd)
        if state == "completed":
            return state
        if state in {"failed", "cancelled", "timeout"}:
            raise SlurmError(f"Slurm job {job_id} ended in state: {state}")
        if state == "missing":
            missing_count += 1
            if missing_count >= missing_confirmations:
                raise SlurmError(f"Slurm job {job_id} disappeared before a terminal accounting state was visible")
        else:
            missing_count = 0

        time.sleep(poke_interval)
