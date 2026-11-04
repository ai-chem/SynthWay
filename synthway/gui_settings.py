"""Validate editable desktop controls before any provider requests."""
import math
from pathlib import Path
import re
import tempfile
import time


def new_run_directory(base, composition):
    """Reserve a unique directory even for simultaneous runs of the same target."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", composition)[:80] or "target"
    return tempfile.mkdtemp(prefix=f"gui_{safe}_{time.strftime('%Y%m%d_%H%M%S')}_", dir=base)


def parse_run_settings(values):
    def integer(key, label, low, high=None):
        raw = values.get(key)
        text = str(raw).strip()
        if not text.isascii() or not text.isdecimal():
            raise ValueError(f"{label} must be a whole number.")
        value = int(text)
        if value < low or (high is not None and value > high):
            limit = f"{low}–{high}" if high is not None else f"at least {low}"
            raise ValueError(f"{label} must be {limit}.")
        return value

    num_routes = integer("-NUM-", "Num routes", 1, 5)
    rpm_limit = integer("-RPM-", "RPM limit", 1)
    max_attempts = integer("-MAXTRY-", "Max tries", 1, 3)
    try:
        raw = values.get("-THR-")
        if isinstance(raw, bool):
            raise ValueError
        threshold = float(str(raw).strip().replace(",", "."))
    except (TypeError, ValueError):
        raise ValueError("Threshold must be a finite number from 0 to 1.") from None
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number from 0 to 1.")
    return num_routes, threshold, rpm_limit, max_attempts
