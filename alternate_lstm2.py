from __future__ import annotations

"""
LSTM_model_v2.py — targeted improvements for the velocity (V) condition.

Changes from LSTM_model_improved.py
────────────────────────────────────
FEATURES
  1. Bandpass-filtered signal channels (requires scipy):
       e_vlo / u_vlo  — very-low-frequency (0.05–0.5 Hz) component of e/u.
                         This is the vestibular sensitivity range; pilots'
                         control strategy *at these frequencies* changes the
                         most between motion-on and motion-off for V.
       e_lo  / u_lo   — low-frequency (0.5–3 Hz) component of e/u.
                         Captures crossover-frequency dynamics for V and P.
     If scipy is unavailable these channels are replaced with zeros and
     excluded from the candidate input combinations.

  2. u_de = mean_centered(u) × derivative(e)
     This product directly encodes pilot lead generation.  In velocity
     control without motion the pilot differentiates the error signal more
     aggressively; the cross-correlation u·de is strongly positive.  With
     motion cueing present the lead requirement falls and u·de decreases.
     This is the single most discriminative feature for the V condition.

  3. e_du = mean_centered(e) × derivative(u)  (complementary to u_de)

  4. eu amplitude ratio preserved — eu is now normalised by the product of
     run-level stds (correlation-like) instead of z-scored, so it retains
     the *relative* amplitude information that z-scoring would destroy.

ARCHITECTURE  (MultiScaleCNNBiLSTMClassifier)
  5. Parallel dilated CNN replaces the single-scale CNN front-end:
       Fast path   — kernel 5 (local patterns, 0.05 s at 100 Hz)
       Medium path — kernel 21 (0.21 s)
       Slow path   — 3 stacked dilated Conv1d with dilation 2→4→8,
                     kernel 7 each.  Total receptive field ≈ 169 samples
                     (1.7 s at 100 Hz) — enough to see a full 0.6 Hz cycle.
     The three path outputs are concatenated then fused with two more
     Conv1d layers.

  6. Window sizes extended to (128, 256, 512, 1024).  A 1024-sample
     window at 100 Hz = 10.24 s, resolving frequencies down to 0.1 Hz.

  Everything else (nested CV, lazy dataset, I/O, progress tracking) is
  unchanged from the previous version.
"""

import copy
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from scipy import stats  # type: ignore
    HAS_SCIPY = True
except Exception:
    stats = None
    HAS_SCIPY = False

try:
    from scipy.signal import butter, filtfilt  # type: ignore
    HAS_SCIPY_SIGNAL = True
except Exception:
    HAS_SCIPY_SIGNAL = False

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

FILENAME_HELP = "ae2224I_measurement_data_subj<1-6>_C<1-6>.npz"
_DEFAULT_FS: float = 100.0   # Hz — assumed when no time vector present


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dir: Optional[str] = None
    save_dir: Optional[str] = None

    # FIX 6: include 1024-sample windows (10.2 s @ 100 Hz)
    window_sizes: Tuple[int, ...] = (128, 256, 512, 1024)
    stride_fraction: float = 0.5

    # FIX 1+2+3: new channels (e_vlo, u_vlo, e_lo, u_lo, u_de, e_du)
    # Candidates with bandpass channels are included; they reduce to zeros
    # gracefully if scipy is unavailable.
    input_combinations: Tuple[Tuple[str, ...], ...] = (
        # ── baselines ─────────────────────────────────────────────────────
        ("e", "u"),
        ("e", "u", "de", "du"),
        ("e", "u", "de", "du", "d2e", "d2u"),
        # ── spectral-band channels (key for V) ────────────────────────────
        ("e", "u", "e_vlo", "u_vlo", "e_lo", "u_lo"),
        ("e", "u", "de", "du", "e_vlo", "u_vlo", "e_lo", "u_lo"),
        # ── pilot-lead cross features (key for V) ─────────────────────────
        ("e", "u", "eu", "u_de", "e_du", "de", "du"),
        ("e", "u", "eu", "u_de", "e_du", "de", "du", "d2e", "d2u"),
        # ── spectral + cross features combined ────────────────────────────
        ("e", "u", "eu", "u_de", "e_du", "de", "du", "e_vlo", "u_vlo", "e_lo", "u_lo"),
        # ── full feature set ──────────────────────────────────────────────
        ("e", "u", "eu", "u_de", "e_du", "de", "du", "d2e", "d2u",
         "e_vlo", "u_vlo", "e_lo", "u_lo", "e_logstd", "u_logstd"),
    )

    batch_size: int = 256
    eval_batch_size: int = 2048
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 8
    val_check_interval: int = 2

    stage1_enabled: bool = True
    stage1_epochs: int = 8
    stage1_patience: int = 3
    stage1_top_k: int = 4
    global_tune_once_per_vehicle: bool = True

    random_state: int = 42
    min_run_length: int = 256   # raised; shorter runs lack low-freq content

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    heartbeat_seconds: float = 10.0
    batch_update_interval: int = 50

    prefer_cuda: bool = True
    use_amp: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    compile_model: bool = False

    default_fs: float = _DEFAULT_FS   # assumed sample rate when no time vector

    filename_pattern: str = (
        r"^ae2224I_measurement_data_subj(?P<subject>[1-6])_(?P<condition>C[1-6])\.npz$"
    )
    signal_keys: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "e": ("e",),
            "u": ("u",),
            "time": ("t", "time"),
        }
    )
    condition_to_label: Dict[str, int] = field(
        default_factory=lambda: {
            "C1": 0, "C2": 0, "C3": 0,
            "C4": 1, "C5": 1, "C6": 1,
        }
    )
    condition_to_vehicle: Dict[str, str] = field(
        default_factory=lambda: {
            "C1": "P", "C2": "V", "C3": "A",
            "C4": "P", "C5": "V", "C6": "A",
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Progress tracker (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, total_trainings: int, heartbeat_seconds: float = 10.0):
        self.total_trainings = max(1, int(total_trainings))
        self.completed_trainings = 0
        self.start_time = time.time()
        self.current_label = ""
        self.current_planned_epochs = 1
        self.current_epoch = 0
        self.current_batch = 0
        self.current_total_batches = 1
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self.last_print_time = 0.0

    def start_training(self, label: str, planned_epochs: int) -> None:
        self.current_label = label
        self.current_planned_epochs = max(1, int(planned_epochs))
        self.current_epoch = 0; self.current_batch = 0
        self.current_total_batches = 1; self.last_print_time = 0.0
        self._print_status("START", newline=True)

    def start_epoch(self, epoch: int, total_batches: int) -> None:
        self.current_epoch = max(1, int(epoch))
        self.current_batch = 0
        self.current_total_batches = max(1, int(total_batches))
        self._maybe_print("RUN", force=True)

    def update_batch(self, batch_idx: int, total_batches: int) -> None:
        self.current_batch = max(0, int(batch_idx))
        self.current_total_batches = max(1, int(total_batches))
        self._maybe_print("RUN", force=False)

    def finish_epoch(self, epoch: int) -> None:
        self.current_epoch = max(1, int(epoch))
        self.current_batch = self.current_total_batches
        self._print_status("RUN", newline=True)

    def finish_training(self, actual_epochs: int) -> None:
        self.current_epoch = max(0, int(actual_epochs))
        self.current_batch = self.current_total_batches
        self.completed_trainings += 1
        self._print_status("DONE", newline=True)

    def _fraction_complete(self) -> float:
        bf = min(1.0, self.current_batch / max(1, self.current_total_batches))
        ef = 0.0
        if self.current_planned_epochs > 0 and self.current_epoch > 0:
            ef = ((self.current_epoch - 1) + bf) / self.current_planned_epochs
        ef = min(1.0, max(0.0, ef))
        return min(1.0, (self.completed_trainings + ef) / self.total_trainings)

    @staticmethod
    def _fmt(s: float) -> str:
        if not np.isfinite(s) or s < 0: return "--:--:--"
        s = int(round(s)); h, r = divmod(s, 3600); m, ss = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{ss:02d}"

    def _status_message(self, prefix: str) -> str:
        elapsed = time.time() - self.start_time
        frac = self._fraction_complete()
        eta = (elapsed / frac - elapsed) if frac > 1e-9 else float("inf")
        return (
            f"[{prefix}] training {min(self.completed_trainings+1, self.total_trainings)}/"
            f"{self.total_trainings} | {frac*100:6.2f}% | "
            f"ep {self.current_epoch}/{self.current_planned_epochs} | "
            f"batch {self.current_batch}/{self.current_total_batches} | "
            f"elapsed {self._fmt(elapsed)} | ETA {self._fmt(eta)} | {self.current_label}"
        )

    def _print_status(self, prefix: str, newline: bool) -> None:
        msg = self._status_message(prefix)
        if newline: print(msg, flush=True)
        else: print(msg, end="\r", flush=True)
        self.last_print_time = time.time()

    def _maybe_print(self, prefix: str, force: bool = False) -> None:
        now = time.time()
        if force or (now - self.last_print_time >= self.heartbeat_seconds):
            self._print_status(prefix, newline=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def configure_torch(config: TrainConfig) -> torch.device:
    use_cuda = bool(config.prefer_cuda and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    if use_cuda:
        if config.allow_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception: pass
        try: torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
        except Exception: pass
    return device


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        if shell is None: return False
        return shell.__class__.__name__ in {"ZMQInteractiveShell", "TerminalInteractiveShell"} or "ipykernel" in sys.modules
    except Exception:
        return "ipykernel" in sys.modules


def sanitize_runtime_config(config: TrainConfig, device: torch.device) -> TrainConfig:
    cfg = copy.deepcopy(config)
    interactive = running_in_notebook()
    windows_platform = sys.platform.startswith("win")
    if device.type != "cuda":
        cfg.use_amp = False; cfg.pin_memory = False
        cfg.persistent_workers = False; cfg.num_workers = 0
        if cfg.eval_batch_size < cfg.batch_size: cfg.eval_batch_size = cfg.batch_size
        return cfg
    if cfg.num_workers < 0: cfg.num_workers = 0
    cpu_count = os.cpu_count() or 4
    if cfg.num_workers == 0: cfg.num_workers = min(8, max(2, cpu_count // 2))
    if interactive and windows_platform:
        cfg.num_workers = 0; cfg.persistent_workers = False; cfg.prefetch_factor = 2
    else:
        cfg.pin_memory = True
        if cfg.num_workers == 0: cfg.persistent_workers = False
    if cfg.eval_batch_size < cfg.batch_size: cfg.eval_batch_size = cfg.batch_size * 2
    return cfg


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0: return float("nan")
    return float(np.mean(y_true == y_pred))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] = (0, 1)) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    labels = list(labels); index = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in index and yp in index: cm[index[yt], index[yp]] += 1
    return cm


def classification_report_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] = (0, 1)) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    report: Dict[str, Dict[str, float]] = {}
    supports: List[int] = []; precisions: List[float] = []; recalls: List[float] = []; f1s: List[float] = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        report[str(label)] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
        supports.append(support); precisions.append(precision); recalls.append(recall); f1s.append(f1)
    weights = np.asarray(supports, dtype=float)
    total_support = int(np.sum(supports))
    weight_sum = float(np.sum(weights)) if float(np.sum(weights)) > 0 else 1.0
    report["accuracy"] = {"value": accuracy_score_np(y_true, y_pred)}
    report["macro avg"] = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1s)) if f1s else 0.0,
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": float(np.sum(np.asarray(precisions) * weights) / weight_sum),
        "recall": float(np.sum(np.asarray(recalls) * weights) / weight_sum),
        "f1-score": float(np.sum(np.asarray(f1s) * weights) / weight_sum),
        "support": total_support,
    }
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Signal processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x)); sigma = float(np.std(x))
    if sigma < 1e-8: return x - mu
    return (x - mu) / sigma


def mean_center(x: np.ndarray) -> np.ndarray:
    """Subtract mean only — preserves amplitude differences between conditions."""
    x = np.asarray(x, dtype=np.float32)
    return x - float(np.mean(x))


def derivative(x: np.ndarray, time: Optional[np.ndarray] = None, fs: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if time is not None:
        time = np.asarray(time, dtype=np.float32).reshape(-1)
        if len(time) != len(x): raise ValueError("time length mismatch")
        return np.gradient(x, time).astype(np.float32)
    dt = 1.0 / fs if fs not in (None, 0) else 1.0
    return np.gradient(x, dt).astype(np.float32)


def _bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    """
    FIX 1: Zero-phase Butterworth bandpass.
    Returns zeros silently if scipy.signal unavailable or filter fails.
    The caller should check HAS_SCIPY_SIGNAL before including these channels
    in input_combinations that matter; they degrade to zero otherwise.
    """
    if not HAS_SCIPY_SIGNAL:
        return np.zeros_like(x)
    nyq = 0.5 * fs
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 1.0 - 1e-4)
    if lo >= hi or len(x) < 20:
        return np.zeros_like(x)
    try:
        b, a = butter(order, [lo, hi], btype="band")
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        return np.zeros_like(x)


def normalize_run_signals(run: Dict[str, Any], default_fs: float = _DEFAULT_FS) -> Dict[str, Any]:
    """
    Compute all feature channels and attach them to the run dict.

    Channels produced
    -----------------
    e, u         : mean-centred (amplitude preserved)
    de, du       : first derivative, z-scored
    d2e, d2u     : second derivative, z-scored
    eu           : normalised cross-product  e·u / (std_e·std_u)
                   — correlation-like, retains relative amplitude ratio
    u_de         : FIX 2 — u × de  (pilot lead generation feature)
    e_du         : FIX 3 — e × du  (complementary plant feature)
    e_vlo, u_vlo : FIX 1 — bandpass 0.05–0.5 Hz (vestibular range)
    e_lo,  u_lo  : FIX 1 — bandpass 0.5–3.0 Hz  (crossover range)
    e_logstd,    : log(std) of raw e/u broadcast as constant channel
    u_logstd
    """
    out = copy.deepcopy(run)
    raw_e = np.asarray(out["e"], dtype=np.float32)
    raw_u = np.asarray(out["u"], dtype=np.float32)
    time_vec = out.get("time")
    fs = float(out.get("fs", default_fs))

    # ── mean-centre (not z-score) — amplitude kept ────────────────────────
    e_c = mean_center(raw_e)
    u_c = mean_center(raw_u)
    out["e"] = e_c
    out["u"] = u_c

    # ── first derivatives ─────────────────────────────────────────────────
    de_raw = derivative(raw_e, time=time_vec, fs=fs)
    du_raw = derivative(raw_u, time=time_vec, fs=fs)
    out["de"] = safe_zscore(de_raw)
    out["du"] = safe_zscore(du_raw)

    # ── second derivatives ────────────────────────────────────────────────
    out["d2e"] = safe_zscore(derivative(de_raw, time=time_vec, fs=fs))
    out["d2u"] = safe_zscore(derivative(du_raw, time=time_vec, fs=fs))

    # ── FIX 4: amplitude-preserving eu (no z-score) ───────────────────────
    std_e = float(np.std(e_c)) + 1e-8
    std_u = float(np.std(u_c)) + 1e-8
    out["eu"] = (e_c * u_c / (std_e * std_u)).astype(np.float32)

    # ── FIX 2: u·de — pilot lead generation feature ───────────────────────
    # Positive when pilot is differentiating (applying lead compensation).
    # For V without motion, this is systematically higher.
    out["u_de"] = safe_zscore(u_c * de_raw)

    # ── FIX 3: e·du — plant response feature ─────────────────────────────
    out["e_du"] = safe_zscore(e_c * du_raw)

    # ── FIX 1: bandpass channels ──────────────────────────────────────────
    # Very low frequency: 0.05–0.5 Hz (vestibular sensitivity range)
    # A 0.5 Hz signal at 100 Hz has a period of 200 samples; windows must
    # be at least 256 samples to contain even one cycle.
    out["e_vlo"] = safe_zscore(_bandpass(raw_e, fs, low_hz=0.05, high_hz=0.5, order=3))
    out["u_vlo"] = safe_zscore(_bandpass(raw_u, fs, low_hz=0.05, high_hz=0.5, order=3))

    # Low frequency: 0.5–3.0 Hz (crossover frequency range for V and P)
    out["e_lo"] = safe_zscore(_bandpass(raw_e, fs, low_hz=0.5, high_hz=3.0, order=3))
    out["u_lo"] = safe_zscore(_bandpass(raw_u, fs, low_hz=0.5, high_hz=3.0, order=3))

    # ── log-std amplitude ─────────────────────────────────────────────────
    n = len(raw_e)
    out["e_logstd"] = np.full(n, float(np.log1p(np.std(raw_e) + 1e-8)), dtype=np.float32)
    out["u_logstd"] = np.full(n, float(np.log1p(np.std(raw_u) + 1e-8)), dtype=np.float32)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (unchanged except normalize_run_signals call)
# ──────────────────────────────────────────────────────────────────────────────

def _first_available_key(npz_obj: Any, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for key in candidates:
        if key in npz_obj.files: return key
    if required: raise KeyError(f"None of {tuple(candidates)} found. Available: {npz_obj.files}")
    return None


def _to_1d_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1: arr = np.asarray(arr.reshape(-1)[0])
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 0: arr = arr.reshape(1)
    return arr.reshape(-1)


def _split_npz_field_into_runs(value: Any) -> List[np.ndarray]:
    arr = np.asarray(value)
    if arr.dtype == object: return [_to_1d_float_array(item) for item in arr.reshape(-1)]
    if arr.ndim <= 1: return [_to_1d_float_array(arr)]
    if arr.ndim == 2 and 1 in arr.shape:
        flat = arr.reshape(-1)
        if flat.dtype == object: return [_to_1d_float_array(item) for item in flat]
        return [_to_1d_float_array(arr)]
    if arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]: return [_to_1d_float_array(arr[i, :]) for i in range(arr.shape[0])]
        return [_to_1d_float_array(arr[:, i]) for i in range(arr.shape[1])]
    return [_to_1d_float_array(arr)]


def _infer_fs(time_vector: Optional[np.ndarray]) -> Optional[float]:
    if time_vector is None or len(time_vector) < 2: return None
    dt = np.diff(np.asarray(time_vector, dtype=np.float32))
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0: return None
    median_dt = float(np.median(dt))
    if abs(median_dt) < 1e-12: return None
    return float(1.0 / median_dt)


def resolve_data_dir(config: TrainConfig, script_dir: Path) -> Path:
    candidates: List[Path] = []
    if config.data_dir not in (None, ""):
        p = Path(config.data_dir)
        candidates.append(p if p.is_absolute() else script_dir / p)
        candidates.append(p if p.is_absolute() else Path.cwd() / p)
    candidates.extend([
        script_dir / "data" / "python_data", script_dir / "python_data",
        Path.cwd() / "data" / "python_data", Path.cwd() / "python_data",
        script_dir.parent / "data" / "python_data", script_dir.parent / "python_data",
    ])
    unique: List[Path] = []; seen: set = set()
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen: seen.add(key); unique.append(c)
    for c in unique:
        if c.exists() and c.is_dir(): return c
    raise FileNotFoundError(f"Data directory not found. Checked:\n" + "\n".join(str(p) for p in unique))


def resolve_save_dir(config: TrainConfig, script_dir: Path) -> Path:
    if config.save_dir in (None, ""):
        return script_dir / "results_lstm_v2"
    p = Path(config.save_dir)
    return p if p.is_absolute() else script_dir / p


def load_runs_from_npz_dir(data_dir, config: TrainConfig, min_run_length: int = 1) -> List[Dict[str, Any]]:
    data_path = Path(data_dir)
    if not data_path.exists(): raise FileNotFoundError(f"Data dir not found: {data_path}")
    pattern = re.compile(config.filename_pattern, re.IGNORECASE)
    npz_files = [p for p in sorted(data_path.glob("*.npz")) if pattern.match(p.name)]
    if not npz_files: raise FileNotFoundError(f"No matching .npz files in {data_path}")

    all_runs: List[Dict[str, Any]] = []
    for npz_file in npz_files:
        match = pattern.match(npz_file.name)
        if match is None: continue
        subject = f"subj{match.group('subject')}"
        condition = match.group("condition").upper()
        label = int(config.condition_to_label[condition])
        vehicle = str(config.condition_to_vehicle[condition])

        with np.load(npz_file, allow_pickle=True) as npz_obj:
            e_key = _first_available_key(npz_obj, config.signal_keys["e"])
            u_key = _first_available_key(npz_obj, config.signal_keys["u"])
            t_key = _first_available_key(npz_obj, config.signal_keys["time"], required=False)
            e_runs = _split_npz_field_into_runs(npz_obj[e_key])
            u_runs = _split_npz_field_into_runs(npz_obj[u_key])
            if len(e_runs) != len(u_runs): raise ValueError(f"{npz_file.name}: e/u count mismatch")
            if t_key is None:
                t_runs: List[Optional[np.ndarray]] = [None] * len(e_runs)
            else:
                raw_t = _split_npz_field_into_runs(npz_obj[t_key])
                if len(raw_t) == 1 and len(e_runs) > 1: t_runs = [raw_t[0]] * len(e_runs)
                elif len(raw_t) == len(e_runs): t_runs = list(raw_t)
                else: raise ValueError(f"{npz_file.name}: t count mismatch")

            for rep_idx, (e_run, u_run, t_run) in enumerate(zip(e_runs, u_runs, t_runs), start=1):
                lengths = [len(e_run), len(u_run)]
                if t_run is not None: lengths.append(len(t_run))
                n = min(lengths)
                if n < min_run_length: continue
                run: Dict[str, Any] = {
                    "e": np.asarray(e_run[:n], dtype=np.float32),
                    "u": np.asarray(u_run[:n], dtype=np.float32),
                    "label": label, "vehicle_type": vehicle, "pilot_id": subject,
                    "condition_id": condition, "repetition_id": f"rep{rep_idx}",
                    "source_file": npz_file.name,
                    "run_id": f"{npz_file.stem}_rep{rep_idx}",
                }
                if t_run is not None:
                    run["time"] = np.asarray(t_run[:n], dtype=np.float32)
                    fs = _infer_fs(run["time"])
                    if fs is not None: run["fs"] = fs
                all_runs.append(normalize_run_signals(run, default_fs=config.default_fs))

    if not all_runs: raise ValueError("No valid runs loaded.")
    return all_runs


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class LazyWindowDataset(Dataset):
    def __init__(self, run_lookup, rows, input_vars):
        self.run_lookup = run_lookup
        self.rows = list(rows)
        self.input_vars = tuple(input_vars)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        run = self.run_lookup[row["run_id"]]
        start, end = int(row["start_idx"]), int(row["end_idx"])
        x = np.stack([run[var][start:end] for var in self.input_vars], axis=-1).astype(np.float32)
        meta = {
            "vehicle_type": row["vehicle_type"], "pilot_id": row["pilot_id"],
            "repetition_id": row["repetition_id"], "run_id": row["run_id"],
            "source_file": row["source_file"], "start_idx": start, "end_idx": end,
        }
        return torch.from_numpy(x), torch.tensor(int(row["y"]), dtype=torch.long), meta


# ──────────────────────────────────────────────────────────────────────────────
# FIX 5: Multi-scale dilated CNN + BiLSTM + Attention
# ──────────────────────────────────────────────────────────────────────────────

class DilatedSlowPath(nn.Module):
    """
    Three stacked dilated Conv1d with dilation 2→4→8, kernel_size=7 each.
    Total receptive field = 6*2 + 6*4 + 6*8 + 1 = 169 samples ≈ 1.7 s @ 100 Hz.
    This captures patterns at 0.6–10 Hz that a non-dilated CNN misses entirely.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=7, padding=6,  dilation=2),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid,        mid, kernel_size=7, padding=12, dilation=4),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid,        mid, kernel_size=7, padding=24, dilation=8),
            nn.BatchNorm1d(mid), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(lstm_out), dim=1)   # (B, L, 1)
        return (weights * lstm_out).sum(dim=1)                  # (B, H)


class MultiScaleCNNBiLSTMClassifier(nn.Module):
    """
    Architecture for v2:

    Input (B, L, C)
      │
      ├─ Fast path   Conv1d k=5            → (B, ch//3, L)
      ├─ Medium path Conv1d k=21           → (B, ch//3, L)
      └─ Slow path   DilatedSlowPath RF≈1.7s → (B, ch-2*(ch//3), L)
              ↓ concat + 2×Conv1d fusion
    CNN out (B, ch, L) → transpose → (B, L, ch)
              ↓
    BiLSTM  (B, L, H*2)
              ↓
    Attention-weighted context (B, H*2)
              + residual (mean-pool CNN → project)
              ↓
    LayerNorm → Linear → GELU → Dropout → Linear → logits (B, 2)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, num_classes: int = 2, cnn_channels: int = 128):
        super().__init__()

        c1 = cnn_channels // 3
        c2 = cnn_channels // 3
        c3 = cnn_channels - c1 - c2

        # ── parallel CNN paths ────────────────────────────────────────────
        self.path_fast = nn.Sequential(
            nn.Conv1d(input_size, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1), nn.GELU(),
        )
        self.path_mid = nn.Sequential(
            nn.Conv1d(input_size, c2, kernel_size=21, padding=10),
            nn.BatchNorm1d(c2), nn.GELU(),
        )
        self.path_slow = DilatedSlowPath(input_size, c3)   # RF ≈ 1.7 s

        # ── fusion convolutions ───────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.GELU(),
        )

        # ── BiLSTM ────────────────────────────────────────────────────────
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=hidden_size,
            num_layers=num_layers, dropout=effective_dropout,
            batch_first=True, bidirectional=True,
        )
        lstm_out_size = hidden_size * 2

        # ── attention + residual ──────────────────────────────────────────
        self.attention = TemporalAttention(lstm_out_size)
        self.residual_proj = nn.Linear(cnn_channels, lstm_out_size)

        # ── classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        xT = x.permute(0, 2, 1)                                    # (B, C, L)
        cnn_out = self.fusion(
            torch.cat([self.path_fast(xT), self.path_mid(xT), self.path_slow(xT)], dim=1)
        )                                                           # (B, ch, L)
        cnn_seq = cnn_out.permute(0, 2, 1)                         # (B, L, ch)
        lstm_out, _ = self.lstm(cnn_seq)                           # (B, L, H*2)
        context = self.attention(lstm_out)                         # (B, H*2)
        residual = self.residual_proj(cnn_seq.mean(dim=1))        # (B, H*2)
        return self.classifier(context + residual)


def build_model(config: TrainConfig, input_size: int, device: torch.device) -> nn.Module:
    cnn_channels = min(192, max(64, config.hidden_size))
    model = MultiScaleCNNBiLSTMClassifier(
        input_size=input_size, hidden_size=config.hidden_size,
        num_layers=config.num_layers, dropout=config.dropout,
        cnn_channels=cnn_channels,
    ).to(device)
    if bool(config.compile_model) and hasattr(torch, "compile"):
        try: model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception: pass
    return model


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: Sequence[Any]):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def make_loader(rows, run_lookup, input_vars, batch_size, shuffle, config, device):
    ds = LazyWindowDataset(run_lookup, rows, input_vars)
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size, "shuffle": shuffle,
        "num_workers": int(config.num_workers), "collate_fn": collate_fn,
        "pin_memory": bool(config.pin_memory and device.type == "cuda"),
        "drop_last": False,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = bool(config.persistent_workers)
        kwargs["prefetch_factor"] = int(config.prefetch_factor)
    return DataLoader(ds, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# CV utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_unique_pilot_ids(runs: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(r["pilot_id"]) for r in runs})


def build_candidate_cache(vehicle_runs, config):
    run_lookup = {str(run["run_id"]): run for run in vehicle_runs}
    cache: Dict[Tuple, Dict[str, Any]] = {}
    for input_vars in config.input_combinations:
        for window_size in config.window_sizes:
            stride = max(1, int(round(window_size * config.stride_fraction)))
            rows_by_pilot: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
            total_windows = 0
            for run in vehicle_runs:
                run_id = str(run["run_id"]); pilot_id = str(run["pilot_id"])
                n = len(run["e"])
                if n < window_size: continue
                for start in range(0, n - window_size + 1, stride):
                    rows_by_pilot[pilot_id].append({
                        "run_id": run_id, "pilot_id": pilot_id,
                        "vehicle_type": str(run["vehicle_type"]),
                        "repetition_id": str(run["repetition_id"]),
                        "source_file": str(run["source_file"]),
                        "start_idx": int(start), "end_idx": int(start + window_size),
                        "y": int(run["label"]),
                    })
                    total_windows += 1
            if total_windows > 0:
                cache[(tuple(input_vars), int(window_size))] = {
                    "input_vars": tuple(input_vars), "window_size": int(window_size),
                    "rows_by_pilot": dict(rows_by_pilot), "total_windows": int(total_windows),
                }
    return run_lookup, cache


def collect_rows_for_pilots(candidate_entry, pilot_ids):
    out: List[Dict[str, Any]] = []
    for pid in pilot_ids: out.extend(candidate_entry["rows_by_pilot"].get(str(pid), []))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _should_run_validation(epoch, max_epochs, val_check_interval):
    interval = max(1, int(val_check_interval))
    return epoch == 1 or epoch == max_epochs or (epoch % interval == 0)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp, tracker=None, batch_update_interval=50):
    model.train()
    total_loss = 0.0; total_n = 0
    total_batches = len(loader)
    for batch_idx, (x, y, _) in enumerate(loader, start=1):
        if device.type == "cuda":
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        else:
            x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        bs = int(x.size(0)); total_loss += float(loss.item()) * bs; total_n += bs
        if tracker is not None and (batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, int(batch_update_interval)) == 0):
            tracker.update_batch(batch_idx, total_batches)
    return total_loss / max(1, total_n)


@torch.inference_mode()
def evaluate_accuracy_only(model, loader, device, use_amp):
    model.eval(); correct = 0; total = 0
    for x, y, _ in loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16): logits = model(x)
            else: logits = model(x)
        else:
            x = x.to(device); y = y.to(device); logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item()); total += int(y.numel())
    return float(correct / total) if total > 0 else float("nan")


@torch.inference_mode()
def predict_loader(model, loader, device, use_amp):
    model.eval()
    y_true: List[int] = []; y_pred: List[int] = []; y_prob: List[float] = []; metas: List[Dict] = []
    for x, y, meta in loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16): logits = model(x)
            else: logits = model(x)
        else:
            x = x.to(device); logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(y.numpy().tolist()); y_pred.extend(preds.tolist())
        y_prob.extend(probs[:, 1].tolist()); metas.extend(meta)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(y_prob, dtype=float), metas


def fit_lstm(train_rows, val_rows, run_lookup, input_vars, config, device, tracker=None,
             training_label="", max_epochs_override=None, patience_override=None,
             val_check_interval_override=None):
    model = build_model(config, input_size=len(input_vars), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    max_epochs = int(max_epochs_override or config.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if device.type == "cuda" else None
    train_loader = make_loader(train_rows, run_lookup, input_vars, config.batch_size, True, config, device)
    val_loader   = make_loader(val_rows,   run_lookup, input_vars, config.eval_batch_size, False, config, device)
    patience = int(patience_override or config.patience)
    val_check_interval = int(val_check_interval_override or config.val_check_interval)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -float("inf"); best_epoch = 1
    history: List[Dict[str, Any]] = []; wait = 0

    if tracker is not None: tracker.start_training(training_label or f"train={len(train_rows)}", max_epochs)

    for epoch in range(1, max_epochs + 1):
        if tracker is not None: tracker.start_epoch(epoch, len(train_loader))
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, config.use_amp,
                                     tracker=tracker, batch_update_interval=config.batch_update_interval)
        scheduler.step()
        val_acc: Optional[float] = None
        if _should_run_validation(epoch, max_epochs, val_check_interval):
            val_acc = evaluate_accuracy_only(model, val_loader, device, config.use_amp)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})
            if val_acc > best_val_acc:
                best_val_acc = val_acc; best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict()); wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if tracker is not None: tracker.finish_epoch(epoch)
                    break
        else:
            history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": None})
        if tracker is not None: tracker.finish_epoch(epoch)

    model.load_state_dict(best_state)
    if tracker is not None: tracker.finish_training(actual_epochs=history[-1]["epoch"] if history else 0)
    return model, history, best_epoch, best_val_acc


def fit_lstm_fixed_epochs(train_rows, run_lookup, input_vars, config, device,
                          num_epochs, tracker=None, training_label=""):
    model = build_model(config, input_size=len(input_vars), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    planned_epochs = max(1, int(num_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=planned_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if device.type == "cuda" else None
    train_loader = make_loader(train_rows, run_lookup, input_vars, config.batch_size, True, config, device)
    history: List[Dict[str, Any]] = []

    if tracker is not None: tracker.start_training(training_label or f"final_train={len(train_rows)}", planned_epochs)

    for epoch in range(1, planned_epochs + 1):
        if tracker is not None: tracker.start_epoch(epoch, len(train_loader))
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, config.use_amp,
                                     tracker=tracker, batch_update_interval=config.batch_update_interval)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss})
        if tracker is not None: tracker.finish_epoch(epoch)

    if tracker is not None: tracker.finish_training(actual_epochs=planned_epochs)
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Prediction & metrics
# ──────────────────────────────────────────────────────────────────────────────

def build_prediction_rows(eval_result):
    rows = []
    for yt, yp, prob, meta in zip(eval_result["y_true"], eval_result["y_pred"],
                                   eval_result["y_prob"], eval_result["metas"]):
        rows.append({
            "true_label": int(yt), "pred_label": int(yp),
            "motion_on_confidence": float(prob),
            "vehicle_type": meta["vehicle_type"], "pilot_id": meta["pilot_id"],
            "repetition_id": meta["repetition_id"], "run_id": meta["run_id"],
            "source_file": meta["source_file"],
            "start_idx": int(meta["start_idx"]), "end_idx": int(meta["end_idx"]),
        })
    return rows


def aggregate_run_predictions(pred_rows):
    grouped: DefaultDict = defaultdict(list)
    for row in pred_rows:
        key = (str(row["vehicle_type"]), str(row["pilot_id"]), str(row["repetition_id"]),
               str(row["run_id"]), int(row["true_label"]))
        grouped[key].append(float(row["motion_on_confidence"]))
    out = []
    for (vehicle_type, pilot_id, repetition_id, run_id, true_label), scores in grouped.items():
        mean_conf = float(np.mean(scores))
        out.append({
            "vehicle_type": vehicle_type, "pilot_id": pilot_id,
            "repetition_id": repetition_id, "run_id": run_id,
            "true_label": true_label, "pred_label": int(mean_conf >= 0.5),
            "motion_on_confidence": mean_conf, "num_windows": int(len(scores)),
        })
    return sorted(out, key=lambda r: (r["vehicle_type"], r["pilot_id"], r["repetition_id"], r["run_id"]))


def compute_metrics_from_prediction_rows(pred_rows):
    y_true = np.asarray([int(row["true_label"]) for row in pred_rows], dtype=int)
    y_pred = np.asarray([int(row["pred_label"]) for row in pred_rows], dtype=int)
    y_prob = np.asarray([float(row["motion_on_confidence"]) for row in pred_rows], dtype=float)
    return {
        "accuracy": accuracy_score_np(y_true, y_pred),
        "confusion_matrix": confusion_matrix_np(y_true, y_pred, labels=(0, 1)),
        "classification_report": classification_report_np(y_true, y_pred, labels=(0, 1)),
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


def run_basic_statistics(run_pred_rows):
    results: Dict[str, Any] = {"aggregated_run_predictions": list(run_pred_rows)}
    if not HAS_SCIPY:
        results["note"] = "scipy not installed; inferential statistics skipped"
        return results
    off_scores = np.asarray([r["motion_on_confidence"] for r in run_pred_rows if int(r["true_label"]) == 0], dtype=float)
    on_scores  = np.asarray([r["motion_on_confidence"] for r in run_pred_rows if int(r["true_label"]) == 1], dtype=float)
    if len(off_scores) >= 2 and len(on_scores) >= 2:
        t_stat, p_val = stats.ttest_ind(off_scores, on_scores, equal_var=False)  # type: ignore[union-attr]
        results["t_test_confidence_off_vs_on"] = {
            "n_off_runs": int(len(off_scores)), "n_on_runs": int(len(on_scores)),
            "mean_off": float(np.mean(off_scores)), "mean_on": float(np.mean(on_scores)),
            "t_statistic": float(t_stat), "p_value": float(p_val),
        }
    return results


def estimate_total_trainings(runs_by_vehicle, config):
    total = 0
    num_candidates = len(config.window_sizes) * len(config.input_combinations)
    stage2_candidates = min(num_candidates, max(1, int(config.stage1_top_k))) if config.stage1_enabled else num_candidates
    for vehicle_runs in runs_by_vehicle.values():
        labels = {int(r["label"]) for r in vehicle_runs}
        pilot_ids = get_unique_pilot_ids(vehicle_runs)
        if len(labels) < 2 or len(pilot_ids) < 3: continue
        n_pilots = len(pilot_ids)
        if bool(config.global_tune_once_per_vehicle):
            search_trainings = n_pilots * (num_candidates + stage2_candidates) if config.stage1_enabled else n_pilots * num_candidates
            total += search_trainings + n_pilots
        else:
            inner_folds = n_pilots - 1
            per_outer = inner_folds * (num_candidates + stage2_candidates) + 1 if config.stage1_enabled else inner_folds * num_candidates + 1
            total += n_pilots * per_outer
    return total


def run_inner_cv_hyperparameter_search(train_pilot_ids, vehicle_type, config, device, run_lookup, candidate_cache, tracker=None):
    train_pilot_ids = list(train_pilot_ids)
    if len(train_pilot_ids) < 2: raise ValueError("Need ≥2 pilots for inner CV.")
    candidate_keys = list(candidate_cache.keys())
    stage1_rows: List[Dict[str, Any]] = []

    if config.stage1_enabled:
        for input_vars, window_size in candidate_keys:
            candidate_entry = candidate_cache[(tuple(input_vars), int(window_size))]
            accs: List[float] = []; epochs: List[int] = []; failed = False
            for val_pilot in train_pilot_ids:
                inner_train = [p for p in train_pilot_ids if p != val_pilot]
                train_rows = collect_rows_for_pilots(candidate_entry, inner_train)
                val_rows   = collect_rows_for_pilots(candidate_entry, [val_pilot])
                if not train_rows or not val_rows or len({int(r["y"]) for r in train_rows}) < 2:
                    failed = True; break
                label = f"{vehicle_type}|s1|val={val_pilot}|vars={','.join(input_vars)}|w={window_size}"
                _, _, be, bva = fit_lstm(train_rows, val_rows, run_lookup, input_vars, config, device,
                                         tracker=tracker, training_label=label,
                                         max_epochs_override=config.stage1_epochs,
                                         patience_override=config.stage1_patience,
                                         val_check_interval_override=max(2, config.val_check_interval))
                accs.append(float(bva)); epochs.append(int(be))
            if not failed and accs:
                stage1_rows.append({
                    "vehicle_type": vehicle_type, "input_vars": ",".join(input_vars),
                    "window_size": int(window_size),
                    "mean_stage1_val_accuracy": float(np.mean(accs)),
                    "std_stage1_val_accuracy": float(np.std(accs, ddof=0)),
                    "mean_stage1_best_epoch": float(np.mean(epochs)),
                })
        if not stage1_rows: raise ValueError("Stage-1 could not evaluate any settings.")
        stage1_rows = sorted(stage1_rows, key=lambda r: (-r["mean_stage1_val_accuracy"], r["std_stage1_val_accuracy"], r["window_size"]))
        selected = stage1_rows[: min(len(stage1_rows), max(1, int(config.stage1_top_k)))]
        selected_keys = [(tuple(str(r["input_vars"]).split(",")), int(r["window_size"])) for r in selected]
    else:
        selected_keys = candidate_keys

    summary_rows: List[Dict[str, Any]] = []; fold_rows: List[Dict[str, Any]] = []
    for input_vars, window_size in selected_keys:
        candidate_entry = candidate_cache.get((tuple(input_vars), int(window_size)))
        if candidate_entry is None: continue
        cand_fold_rows: List[Dict[str, Any]] = []; failed = False
        for val_pilot in train_pilot_ids:
            inner_train = [p for p in train_pilot_ids if p != val_pilot]
            train_rows = collect_rows_for_pilots(candidate_entry, inner_train)
            val_rows   = collect_rows_for_pilots(candidate_entry, [val_pilot])
            if not train_rows or not val_rows or len({int(r["y"]) for r in train_rows}) < 2:
                failed = True; break
            label = f"{vehicle_type}|inner|val={val_pilot}|vars={','.join(input_vars)}|w={window_size}"
            _, _, be, bva = fit_lstm(train_rows, val_rows, run_lookup, input_vars, config, device,
                                     tracker=tracker, training_label=label)
            cand_fold_rows.append({
                "vehicle_type": vehicle_type, "validation_pilot": val_pilot,
                "input_vars": ",".join(input_vars), "window_size": int(window_size),
                "best_epoch": int(be), "best_val_accuracy": float(bva),
            })
        if failed or not cand_fold_rows: continue
        fold_rows.extend(cand_fold_rows)
        accs = [r["best_val_accuracy"] for r in cand_fold_rows]
        epcs = [r["best_epoch"] for r in cand_fold_rows]
        summary_rows.append({
            "vehicle_type": vehicle_type, "input_vars": ",".join(input_vars),
            "window_size": int(window_size),
            "mean_inner_val_accuracy": float(np.mean(accs)),
            "std_inner_val_accuracy": float(np.std(accs, ddof=0)),
            "mean_best_epoch": float(np.mean(epcs)),
            "num_inner_folds": int(len(cand_fold_rows)),
            "total_windows_candidate": int(candidate_entry["total_windows"]),
        })
    if not summary_rows: raise ValueError("Inner CV could not evaluate any settings.")
    summary_rows = sorted(summary_rows, key=lambda r: (-r["mean_inner_val_accuracy"], r["std_inner_val_accuracy"], r["window_size"]))
    return summary_rows[0], summary_rows, fold_rows


def run_vehicle_experiment(runs_for_vehicle, vehicle_type, config, device, tracker=None):
    pilot_ids = get_unique_pilot_ids(runs_for_vehicle)
    if len(pilot_ids) < 3: raise ValueError(f"{vehicle_type}: need ≥3 pilots.")

    outer_fold_rows: List[Dict[str, Any]] = []
    combined_window_rows: List[Dict[str, Any]] = []
    combined_run_rows: List[Dict[str, Any]] = []

    run_lookup_all, candidate_cache_all = build_candidate_cache(runs_for_vehicle, config)

    global_best_input_vars: Optional[Tuple[str, ...]] = None
    global_best_window_size: Optional[int] = None
    global_selected_num_epochs: Optional[int] = None

    if bool(config.global_tune_once_per_vehicle):
        print(f"\nVehicle {vehicle_type}: shared hyperparameter search")
        filtered = {k: v for k, v in candidate_cache_all.items()
                    if all(v["rows_by_pilot"].get(p, []) for p in pilot_ids)}
        if not filtered: raise ValueError(f"{vehicle_type}: no candidate had data for every pilot.")
        best_row, _, _ = run_inner_cv_hyperparameter_search(
            pilot_ids, vehicle_type, config, device, run_lookup_all, filtered, tracker)
        global_best_input_vars = tuple(str(best_row["input_vars"]).split(","))
        global_best_window_size = int(best_row["window_size"])
        global_selected_num_epochs = max(1, int(round(float(best_row["mean_best_epoch"]))))
        print(f"Vehicle {vehicle_type}: selected → w={global_best_window_size}, "
              f"vars={','.join(global_best_input_vars)}, epochs={global_selected_num_epochs}")

    for test_pilot in pilot_ids:
        outer_train_pilots = [p for p in pilot_ids if p != test_pilot]
        outer_train_runs = [r for r in runs_for_vehicle if str(r["pilot_id"]) != str(test_pilot)]
        if len({int(r["label"]) for r in outer_train_runs}) < 2:
            print(f"Skipping {vehicle_type}/{test_pilot}: single class"); continue
        print(f"\nVehicle {vehicle_type}: outer fold test = {test_pilot}")

        if bool(config.global_tune_once_per_vehicle):
            best_input_vars = global_best_input_vars
            best_window_size = global_best_window_size
            selected_num_epochs = global_selected_num_epochs
        else:
            filtered = {k: v for k, v in candidate_cache_all.items()
                        if all(v["rows_by_pilot"].get(p, []) for p in outer_train_pilots + [test_pilot])}
            best_row, _, _ = run_inner_cv_hyperparameter_search(
                outer_train_pilots, vehicle_type, config, device, run_lookup_all, filtered, tracker)
            best_input_vars = tuple(str(best_row["input_vars"]).split(","))
            best_window_size = int(best_row["window_size"])
            selected_num_epochs = max(1, int(round(float(best_row["mean_best_epoch"]))))

        candidate_entry = candidate_cache_all[(best_input_vars, best_window_size)]
        train_rows = collect_rows_for_pilots(candidate_entry, outer_train_pilots)
        test_rows  = collect_rows_for_pilots(candidate_entry, [test_pilot])

        final_model, _ = fit_lstm_fixed_epochs(
            train_rows, run_lookup_all, best_input_vars, config, device,
            num_epochs=selected_num_epochs, tracker=tracker,
            training_label=f"{vehicle_type}|outer-final|test={test_pilot}|w={best_window_size}",
        )
        test_loader = make_loader(test_rows, run_lookup_all, best_input_vars, config.eval_batch_size, False, config, device)
        y_true, y_pred, y_prob, metas = predict_loader(final_model, test_loader, device, config.use_amp)
        window_rows = build_prediction_rows({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "metas": metas})
        for row in window_rows:
            row.update({"outer_test_pilot": test_pilot, "selected_input_vars": ",".join(best_input_vars),
                        "selected_window_size": best_window_size, "selected_num_epochs": selected_num_epochs})
        run_rows = aggregate_run_predictions(window_rows)
        for row in run_rows:
            row.update({"outer_test_pilot": test_pilot, "selected_input_vars": ",".join(best_input_vars),
                        "selected_window_size": best_window_size, "selected_num_epochs": selected_num_epochs})

        run_metrics = compute_metrics_from_prediction_rows(run_rows)
        outer_fold_rows.append({
            "vehicle_type": vehicle_type, "outer_test_pilot": test_pilot,
            "selected_input_vars": ",".join(best_input_vars),
            "selected_window_size": best_window_size, "selected_num_epochs": selected_num_epochs,
            "num_test_runs": int(len(run_rows)), "run_level_accuracy": float(run_metrics["accuracy"]),
        })
        combined_window_rows.extend(window_rows); combined_run_rows.extend(run_rows)

    if not combined_run_rows: raise ValueError(f"No valid outer folds for {vehicle_type}.")
    test_result = compute_metrics_from_prediction_rows(combined_run_rows)
    statistics  = run_basic_statistics(combined_run_rows)
    return {
        "vehicle_type": vehicle_type, "outer_fold_rows": outer_fold_rows,
        "window_prediction_rows": combined_window_rows, "run_prediction_rows": combined_run_rows,
        "test_result": test_result, "statistics": statistics,
        "num_outer_folds": int(len(outer_fold_rows)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O & plotting
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None: path.write_text("", encoding="utf-8"); return
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(fieldnames)).writeheader()
        return
    if fieldnames is None:
        keys: List[str] = []; seen: set = set()
        for row in rows:
            for k in row:
                if k not in seen: seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows: writer.writerow(row)


def _save_plot_confusion_matrix(cm, path, title):
    if plt is None: return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No motion", "Pred: Motion"])
    ax.set_yticklabels(["Actual: No motion", "Actual: Motion"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_plot_pilot_accuracy(outer_fold_rows, path, title):
    if plt is None or not outer_fold_rows: return
    pilots = [str(r["outer_test_pilot"]) for r in outer_fold_rows]
    vals = [float(r["run_level_accuracy"]) for r in outer_fold_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pilots, vals); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title(title)
    for i, v in enumerate(vals): ax.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_plot_overall_accuracy(summary_rows, path):
    if plt is None or not summary_rows: return
    vehicles = [str(r["vehicle_type"]) for r in summary_rows]
    vals = [float(r["run_level_accuracy"]) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(vehicles, vals); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("Run-level accuracy by vehicle type")
    for i, v in enumerate(vals): ax.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_vehicle_results(result, save_dir):
    vehicle_dir = Path(save_dir) / str(result["vehicle_type"])
    vehicle_dir.mkdir(parents=True, exist_ok=True)
    write_csv(vehicle_dir / "pilot_results.csv", result["outer_fold_rows"])
    write_csv(vehicle_dir / "run_predictions.csv", result["run_prediction_rows"])
    cm = result["test_result"]["confusion_matrix"]
    write_csv(vehicle_dir / "confusion_matrix.csv",
              [{"actual": "no_motion", "predicted_no_motion": int(cm[0, 0]), "predicted_motion": int(cm[0, 1])},
               {"actual": "motion",    "predicted_no_motion": int(cm[1, 0]), "predicted_motion": int(cm[1, 1])}],
              fieldnames=["actual", "predicted_no_motion", "predicted_motion"])
    _save_plot_confusion_matrix(cm, vehicle_dir / "confusion_matrix.png",
                                f"Vehicle {result['vehicle_type']} confusion matrix")
    _save_plot_pilot_accuracy(result["outer_fold_rows"], vehicle_dir / "pilot_accuracy.png",
                              f"Vehicle {result['vehicle_type']} accuracy by held-out pilot")
    summary_payload = {
        "vehicle_type": result["vehicle_type"], "outer_folds": result["num_outer_folds"],
        "run_level_accuracy": result["test_result"]["accuracy"],
        "confusion_matrix": result["test_result"]["confusion_matrix"].tolist(),
        "statistics": {k: v for k, v in result["statistics"].items() if k != "aggregated_run_predictions"},
    }
    (vehicle_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    with (vehicle_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Vehicle type: {result['vehicle_type']}\n")
        f.write(f"Outer folds: {result['num_outer_folds']}\n")
        f.write(f"Overall run-level accuracy: {result['test_result']['accuracy']:.3f}\n")
        f.write("Confusion matrix:\n")
        f.write(f"  Actual no motion -> predicted no motion: {int(cm[0, 0])}\n")
        f.write(f"  Actual no motion -> predicted motion   : {int(cm[0, 1])}\n")
        f.write(f"  Actual motion -> predicted no motion   : {int(cm[1, 0])}\n")
        f.write(f"  Actual motion -> predicted motion      : {int(cm[1, 1])}\n")
        if "t_test_confidence_off_vs_on" in result["statistics"]:
            ttest = result["statistics"]["t_test_confidence_off_vs_on"]
            f.write("\nConfidence comparison:\n")
            f.write(f"  Mean no motion confidence: {ttest['mean_off']:.3f}\n")
            f.write(f"  Mean motion confidence   : {ttest['mean_on']:.3f}\n")
            f.write(f"  p-value                  : {ttest['p_value']:.4g}\n")
    return {"vehicle_type": result["vehicle_type"],
            "run_level_accuracy": float(result["test_result"]["accuracy"]),
            "outer_folds": int(result["num_outer_folds"])}


def print_dataset_summary(runs):
    print("=" * 88)
    print(f"Dataset summary | Total valid runs: {len(runs)}")
    by_vehicle = Counter(str(r["vehicle_type"]) for r in runs)
    by_label   = Counter(int(r["label"]) for r in runs)
    print(f"Pilots: {', '.join(sorted({str(r['pilot_id']) for r in runs}))}")
    print(f"Runs by vehicle: {dict(by_vehicle)} | label: {{0: {by_label.get(0,0)}, 1: {by_label.get(1,0)}}}")
    print(f"scipy.signal available for bandpass: {HAS_SCIPY_SIGNAL}")
    if not HAS_SCIPY_SIGNAL:
        print("  ⚠  Install scipy to enable e_vlo / u_vlo / e_lo / u_lo channels (key for V).")
    print("-" * 88)
    print(f"{'Vehicle':<10}{'Runs':>8}{'No motion':>12}{'Motion':>10}{'Pilots':>12}")
    for vehicle in sorted(by_vehicle.keys()):
        vruns = [r for r in runs if str(r["vehicle_type"]) == vehicle]
        lc = Counter(int(r["label"]) for r in vruns)
        pc = len({str(r['pilot_id']) for r in vruns})
        print(f"{vehicle:<10}{len(vruns):>8}{lc.get(0,0):>12}{lc.get(1,0):>10}{pc:>12}")
    print("=" * 88)


def print_compute_summary(device, config):
    print("=" * 88)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
        print(f"AMP: {config.use_amp}  |  TF32: {config.allow_tf32}")
    print(f"Architecture  : MultiScaleCNNBiLSTMClassifier (dilated slow path, RF≈1.7 s)")
    print(f"Window sizes  : {config.window_sizes}")
    print(f"Input combos  : {len(config.input_combinations)}")
    print(f"hidden_size   : {config.hidden_size}")
    print(f"default_fs    : {config.default_fs} Hz (assumed if no time vector)")
    print("=" * 88)


def print_summary(result):
    cm = result["test_result"]["confusion_matrix"]
    print("=" * 88)
    print(f"Vehicle: {result['vehicle_type']}  |  Outer CV folds: {result['num_outer_folds']}")
    print(f"Run-level accuracy: {result['test_result']['accuracy']:.3f}")
    print("Confusion matrix:")
    print(f"  Actual no motion -> no motion: {int(cm[0,0])}   motion: {int(cm[0,1])}")
    print(f"  Actual motion    -> no motion: {int(cm[1,0])}   motion: {int(cm[1,1])}")
    if result["outer_fold_rows"]:
        print(f"{'Pilot':<12}{'Accuracy':>10}{'Window':>8}{'Inputs':>32}{'Epochs':>8}")
        for row in result["outer_fold_rows"]:
            print(f"{str(row['outer_test_pilot']):<12}"
                  f"{float(row['run_level_accuracy']):>10.3f}"
                  f"{int(row['selected_window_size']):>8}"
                  f"{str(row['selected_input_vars']):>32}"
                  f"{int(row['selected_num_epochs']):>8}")
    print("=" * 88)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config: TrainConfig) -> Dict[str, Any]:
    set_seed(config.random_state)
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    data_dir = resolve_data_dir(config, script_dir)
    save_dir = resolve_save_dir(config, script_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = configure_torch(config)
    config = sanitize_runtime_config(config, device)

    print(f"Data : {data_dir}")
    print(f"Save : {save_dir}")
    print_compute_summary(device, config)

    runs = load_runs_from_npz_dir(data_dir, config=config, min_run_length=config.min_run_length)
    print_dataset_summary(runs)

    runs_by_vehicle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs: runs_by_vehicle[str(run["vehicle_type"])].append(run)

    total_trainings = estimate_total_trainings(runs_by_vehicle, config)
    print(f"Estimated model trainings: {total_trainings}")
    tracker = ProgressTracker(total_trainings, heartbeat_seconds=config.heartbeat_seconds)

    overall_rows: List[Dict[str, Any]] = []
    all_results: Dict[str, Any] = {}

    for vehicle_type in sorted(runs_by_vehicle.keys()):
        vehicle_runs = runs_by_vehicle[vehicle_type]
        labels = {int(r["label"]) for r in vehicle_runs}
        pilot_ids = get_unique_pilot_ids(vehicle_runs)
        if len(labels) < 2: print(f"Skipping {vehicle_type}: one class only"); continue
        if len(pilot_ids) < 3: print(f"Skipping {vehicle_type}: <3 pilots"); continue
        result = run_vehicle_experiment(vehicle_runs, vehicle_type, config, device, tracker=tracker)
        all_results[vehicle_type] = result
        overall_rows.append(save_vehicle_results(result, save_dir))
        print_summary(result)

    if overall_rows:
        write_csv(save_dir / "overall_results.csv", overall_rows)
        _save_plot_overall_accuracy(overall_rows, save_dir / "overall_accuracy.png")
        with (save_dir / "overall_results.txt").open("w", encoding="utf-8") as f:
            f.write("Overall run-level accuracy by vehicle type\n" + "=" * 40 + "\n")
            for row in overall_rows:
                f.write(f"{row['vehicle_type']}: {row['run_level_accuracy']:.3f}\n")
    else:
        print("No experiments completed.")
    return all_results


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    config = TrainConfig(
        data_dir=r"C:\\Users\\bramb\\Downloads\\AI_project_simulator\\A02-TAS-Repository\\data\\python_data",
        save_dir=r"C:\\Users\\bramb\\Downloads\\AI_project_simulator\\A02-TAS-Repository\\results_lstm_v2",

        # FIX 6: windows include 1024 (10.2 s @ 100 Hz → resolves 0.1 Hz)
        window_sizes=(128, 256, 512, 1024),
        stride_fraction=0.5,

        # All new input combinations (bandpass + cross-signal features)
        input_combinations=(
            ("e", "u"),
            ("e", "u", "de", "du"),
            ("e", "u", "de", "du", "d2e", "d2u"),
            ("e", "u", "e_vlo", "u_vlo", "e_lo", "u_lo"),
            ("e", "u", "de", "du", "e_vlo", "u_vlo", "e_lo", "u_lo"),
            ("e", "u", "eu", "u_de", "e_du", "de", "du"),
            ("e", "u", "eu", "u_de", "e_du", "de", "du", "d2e", "d2u"),
            ("e", "u", "eu", "u_de", "e_du", "de", "du", "e_vlo", "u_vlo", "e_lo", "u_lo"),
            ("e", "u", "eu", "u_de", "e_du", "de", "du", "d2e", "d2u",
             "e_vlo", "u_vlo", "e_lo", "u_lo", "e_logstd", "u_logstd"),
        ),

        batch_size=256 if use_cuda else 64,
        eval_batch_size=2048 if use_cuda else 256,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=5e-4,
        weight_decay=1e-4,
        max_epochs=40,
        patience=8,
        val_check_interval=2,

        stage1_enabled=True,
        stage1_epochs=8,
        stage1_patience=3,
        stage1_top_k=4,
        global_tune_once_per_vehicle=True,

        random_state=42,
        min_run_length=256,   # raised — very short runs lack low-freq content
        default_fs=100.0,     # ← set to your actual sample rate if known

        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
        prefetch_factor=4,
        prefer_cuda=True,
        use_amp=use_cuda,
        allow_tf32=use_cuda,
        cudnn_benchmark=use_cuda,
        compile_model=False,
    )
    main(config)
