from __future__ import annotations

import ctypes
import shutil
import subprocess
import time


_CACHE: dict[str, object] = {"at": 0.0, "data": {}}
_CACHE_SECONDS = 5.0


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_status() -> dict:
    now = time.monotonic()
    cached = _CACHE.get("data")
    if isinstance(cached, dict) and cached and now - float(_CACHE.get("at", 0.0)) < _CACHE_SECONDS:
        return cached

    data = {
        "cuda": cuda_status(),
        "ram": ram_status(),
        "vram": vram_status(),
    }
    _CACHE["at"] = now
    _CACHE["data"] = data
    return data


def cuda_status() -> dict:
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "label": "CUDA available" if available else "CUDA unavailable",
            "detail": torch.cuda.get_device_name(0) if available else f"torch {torch.__version__}",
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or ""),
        }
    except Exception as exc:
        return {
            "available": False,
            "label": "CUDA unknown",
            "detail": str(exc),
            "torch": "",
            "cuda": "",
        }


def ram_status() -> dict:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
            used = max(0, total - available)
            return memory_payload("RAM", used, total, int(status.dwMemoryLoad))
    except Exception:
        pass
    return {"label": "RAM unknown", "used": None, "total": None, "percent": None, "detail": "RAM status unavailable"}


def vram_status() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
            if line:
                used_text, total_text, util_text, name = [part.strip() for part in line.split(",", 3)]
                used = int(used_text) * 1024 * 1024
                total = int(total_text) * 1024 * 1024
                percent = int(round((used / total) * 100)) if total else 0
                payload = memory_payload("VRAM", used, total, percent)
                payload["detail"] = f"{payload['detail']} - {name} - {util_text}% GPU utilization"
                payload["gpu_utilization"] = int(util_text)
                return payload
        except Exception as exc:
            return {"label": "VRAM unknown", "used": None, "total": None, "percent": None, "detail": str(exc)}

    return torch_vram_status()


def torch_vram_status() -> dict:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {"label": "VRAM unavailable", "used": None, "total": None, "percent": None, "detail": "CUDA is unavailable"}
        free, total = torch.cuda.mem_get_info()
        used = max(0, int(total) - int(free))
        return memory_payload("VRAM", used, int(total), int(round((used / total) * 100)) if total else 0)
    except Exception as exc:
        return {"label": "VRAM unknown", "used": None, "total": None, "percent": None, "detail": str(exc)}


def memory_payload(name: str, used: int, total: int, percent: int) -> dict:
    return {
        "label": f"{name} {format_compact_gb(used)}/{format_compact_gb(total)} used",
        "used": used,
        "total": total,
        "percent": percent,
        "detail": f"{format_bytes(used)} / {format_bytes(total)} used ({percent}%)",
    }


def format_compact_gb(value: int) -> str:
    amount = value / (1024 ** 3)
    if amount >= 10 or amount.is_integer():
        return f"{amount:.0f}GB"
    return f"{amount:.1f}GB"


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"
