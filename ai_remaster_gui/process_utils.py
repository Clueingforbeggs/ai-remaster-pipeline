from __future__ import annotations

import os
import signal
import subprocess


def first_int_after(text: str, marker: str) -> int:
    for line in text.splitlines():
        if marker in line:
            tail = line.split(marker, 1)[1].strip().split()
            if tail:
                try:
                    return int(tail[0].strip(":,"))
                except ValueError:
                    return 0
    return 0


def download_progress_percent(text: str) -> int | None:
    status = download_progress_status(text)
    return int(status["percent"]) if status else None


def download_progress_status(text: str) -> dict[str, str | int] | None:
    latest: dict[str, str | int] | None = None
    marker = "Download progress:"
    for line in text.splitlines():
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].strip()
        value = tail.split("%", 1)[0].strip()
        try:
            percent = max(0, min(100, int(value)))
        except ValueError:
            continue
        status: dict[str, str | int] = {"percent": percent}
        after_percent = tail.split("%", 1)[1].strip() if "%" in tail else ""
        eta_marker = "ETA "
        if eta_marker in after_percent:
            eta = after_percent.split(eta_marker, 1)[1].strip().strip(".,")
            if eta:
                status["eta"] = eta
        latest = status
    return latest


def download_eta_label(status: dict[str, str | int] | None) -> str:
    if status and status.get("eta"):
        return f", ETA {status['eta']}"
    return ""


def outpaint_chunk_progress(text: str) -> dict[str, int]:
    done = count_lines_matching(text, ("Wrote raw Comfy chunk", "Reuse raw Comfy chunk"))
    total = 0
    current = 0
    for line in text.splitlines():
        marker = "Outpaint chunk "
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].split(":", 1)[0]
        if "/" not in tail:
            continue
        try:
            left, right = tail.split("/", 1)
            current = max(current, int(left.strip()))
            total = max(total, int(right.strip()))
        except ValueError:
            pass
    if total:
        current = max(1, min(total, current or min(done + 1, total)))
    return {"done": done, "current": current, "total": total}


def upscale_chunk_progress(text: str) -> dict[str, int]:
    done = count_lines_matching(text, ("Wrote upscaled chunk", "Reuse upscaled chunk"))
    total = 0
    current = 0
    for line in text.splitlines():
        marker = "Upscale chunk "
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].split(":", 1)[0]
        if "/" not in tail:
            continue
        try:
            left, right = tail.split("/", 1)
            current = max(current, int(left.strip()))
            total = max(total, int(right.strip()))
        except ValueError:
            pass
    if total:
        current = max(1, min(total, current or min(done + 1, total)))
    return {"done": done, "current": current, "total": total}


def outpaint_eta_label(elapsed: float, done: int, current: int, total: int) -> str:
    if total <= 0 or done >= total:
        return ""
    if done <= 0:
        return ", ETA calculating"
    average_seconds = elapsed / done
    remaining_seconds = max(0.0, average_seconds * (total - done))
    return f", ETA {format_duration(remaining_seconds)}"

def count_lines_matching(text: str, prefixes: tuple[str, ...]) -> int:
    return sum(1 for line in text.splitlines() if line.startswith(prefixes))

def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()

def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"
