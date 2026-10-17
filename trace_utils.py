"""HDF5 inspection and efficient raw-trace sampling for DNA Trace Explorer.

No Streamlit dependency: can be tested or reused for batch plotting.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SALTS = ("LiCl", "NaCl", "KCl", "RbCl", "CsCl")
SALT_COLORS = {
    "LiCl": "#286CC6", "NaCl": "#0F9E83", "KCl": "#C49A17",
    "RbCl": "#E07835", "CsCl": "#C94666", "Unknown": "#667085",
}


def identify(filename: str) -> dict[str, Any]:
    stem = Path(filename).stem
    salt = next((s for s in SALTS if re.search(rf"(?<![A-Za-z]){s}(?![A-Za-z])", stem, re.I)), "Unknown")
    vol = re.search(r"(\d+(?:\.\d+)?)\s*mV(?![A-Za-z0-9])", stem, re.I)
    if re.search(r"(?<![A-Za-z0-9])(?:1\s*k\s*b(?:p)?|1kbp)(?![A-Za-z0-9])", stem, re.I):
        dna = "1 kb"
    elif re.search(r"lambda|λ", stem, re.I):
        dna = "λ-DNA"
    else:
        dna = "Unknown"
    return {"salt": salt, "dna": dna, "voltage_mV": float(vol.group(1)) if vol else None}


def _read_sampling_rate(f: h5py.File, ds: h5py.Dataset) -> float | None:
    for obj in (ds, ds.parent, f):
        for key in ("sampling_rate", "sample_rate", "sampling_frequency", "fs", "rate", "sampling_rate_hz"):
            if key in obj.attrs:
                val = np.asarray(obj.attrs[key]).ravel()
                if val.size:
                    try:
                        n = float(val[0]);
                        if np.isfinite(n) and n > 0: return n
                    except (TypeError, ValueError):
                        pass
    return None


def inspect_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with h5py.File(p, "r") as f:
        candidates: list[str] = []
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.dtype.kind in "iuf" and obj.size > 0:
                candidates.append(name)
        f.visititems(visit)
        if not candidates:
            raise ValueError("No 1-dimensional numeric dataset found in this HDF5 file.")
        chosen = next((n for n in candidates if n.split("/")[-1] == "selected_data"), None)
        if chosen is None:
            chosen = max(candidates, key=lambda n: f[n].size)
        ds = f[chosen]
        fs = _read_sampling_rate(f, ds)
        meta_dur = f.attrs.get("duration", None)
        try: meta_dur = float(meta_dur) if meta_dur is not None else None
        except (ValueError, TypeError): meta_dur = None
        attrs = {str(k): str(v) for k,v in f.attrs.items()}
        n = int(ds.size)
    parsed = identify(p.name)
    return {
        "path": str(p.resolve()), "name": p.name, "dataset": chosen,
        "n_samples": n, "sampling_rate_hz": fs,
        "signal_duration_s": n / fs if fs else None,
        "metadata_duration_s": meta_dur,
        "size_mb": p.stat().st_size / (1024**2),
        "file_attrs": attrs, **parsed,
    }


def read_envelope(path: str, dataset: str, sampling_rate_hz: float,
                  start_s: float, duration_s: float, max_bins: int = 1800) -> dict[str, np.ndarray]:
    """Streaming min/max/mean per time bin, preserving brief extremes over long windows.

    Bounded buffer of about (128 * samples_per_bin * 8) bytes. Does not load
    a full 50/100-second raw trace into RAM.
    Returned t is RELATIVE to selected region start, in seconds.
    """
    if sampling_rate_hz <= 0 or duration_s <= 0 or max_bins < 2:
        raise ValueError("Sampling rate, duration and max_bins must be positive.")
    with h5py.File(path, "r") as f:
        ds = f[dataset]
        sample_a = max(0, min(ds.size, int(round(start_s * sampling_rate_hz))))
        sample_b = max(sample_a, min(ds.size, int(round((start_s + duration_s) * sampling_rate_hz))))
        n = sample_b - sample_a
        if n <= 0:
            return {k: np.array([]) for k in ("t", "min", "max", "mean")}
        bin_width = max(1, math.ceil(n / max_bins))
        nb = math.ceil(n / bin_width)
        tt = np.empty(nb, dtype=np.float64)
        lower = np.empty(nb, dtype=np.float64)
        upper = np.empty(nb, dtype=np.float64)
        avg = np.empty(nb, dtype=np.float64)
        # Cap memory even for unusually long windows.
        bins_per_block = max(1, min(128, int(1_500_000 / bin_width)))
        for b0 in range(0, nb, bins_per_block):
            bc = min(bins_per_block, nb - b0)
            lo = sample_a + b0 * bin_width
            hi = min(sample_b, lo + bc * bin_width)
            block = ds[lo:hi].astype(np.float64, copy=False)
            if block.size != bc * bin_width:
                block = np.pad(block, (0, bc * bin_width - block.size), constant_values=np.nan)
            matrix = block.reshape(bc, bin_width)
            with np.errstate(invalid="ignore", divide="ignore"):
                lower[b0:b0+bc] = np.nanmin(matrix, axis=1)
                upper[b0:b0+bc] = np.nanmax(matrix, axis=1)
                avg[b0:b0+bc] = np.nanmean(matrix, axis=1)
            centers = (np.arange(b0,b0+bc,dtype=float) + 0.5) * bin_width
            tt[b0:b0+bc] = np.minimum(centers, n-0.5) / sampling_rate_hz
        return {"t": tt, "min": lower, "max": upper, "mean": avg}


def read_raw_window(path: str, dataset: str, sampling_rate_hz: float,
                    start_s: float, duration_ms: float,
                    max_samples: int = 300_000) -> dict[str, np.ndarray]:
    if sampling_rate_hz <= 0 or duration_ms <= 0:
        raise ValueError("Sampling rate and duration must be positive.")
    with h5py.File(path,"r") as f:
        ds = f[dataset]
        a = max(0, min(ds.size, int(round(start_s * sampling_rate_hz))))
        b = max(a, min(ds.size, int(round((start_s + duration_ms/1000) * sampling_rate_hz))))
        if b-a > max_samples:
            raise ValueError("Raw zoom exceeds 300,000 samples. Select a shorter time range.")
        yy = ds[a:b].astype(float, copy=False)
    xx = np.arange(yy.size, dtype=float) * 1000 / sampling_rate_hz
    return {"t_ms": xx, "current": yy}
