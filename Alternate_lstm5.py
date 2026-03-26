from __future__ import annotations

"""
LSTM_model_v5.py

Changes from v4 that fix the 0.50 cliff and the A-vehicle bias flip
════════════════════════════════════════════════════════════════════

FIX 1 — PER-OUTER-FOLD THRESHOLD CALIBRATION  (fixes the 0.50 cliff)
  In v3/v4, a single global threshold was calibrated once from the inner
  CV and then applied identically to every test pilot.  When the model
  produces different confidence levels for different pilots (which happens
  even with class-balanced training), a shared threshold can land perfectly
  between two pilot groups: all of one group are classified identically
  wrong → 0.50 accuracy; all of another group are classified identically
  right → 1.00 accuracy.

  Fix: after the final model is fitted on the outer-training pilots, run
  it over a dedicated calibration split (a random 30 % stratified hold-out
  of the TRAINING pilots' windows), find the threshold that maximises
  Youden's J on those predictions, and apply THAT threshold to the test
  pilot.  Every outer fold now has its own threshold tuned to the specific
  model weights used for that fold.

FIX 2 — CAPPED CLASS WEIGHTS  (fixes the A bias flip)
  v4's raw inverse-frequency weights can exceed 2× for a fold that happens
  to have a 60/40 split, causing the model to over-correct and create a
  no-motion bias.  Weights are now clamped to [0.5, 2.0].

FIX 3 — GLOBAL THRESHOLD REMOVED FROM INNER CV
  The inner CV no longer tries to calibrate a global threshold that will
  be shared across all outer folds.  The inner CV is used only for
  hyperparameter selection (window size, input features, num epochs).
  Threshold calibration happens once per outer fold on training data.

Everything else (architecture, features, nested CV structure, progress
tracking, I/O) is unchanged from v4.
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

try:
    from scipy import stats as scipy_stats   # type: ignore
    HAS_SCIPY = True
except Exception:
    scipy_stats = None; HAS_SCIPY = False

try:
    from scipy.signal import butter, filtfilt  # type: ignore
    HAS_SCIPY_SIGNAL = True
except Exception:
    HAS_SCIPY_SIGNAL = False

try:
    import matplotlib.pyplot as plt   # type: ignore
except Exception:
    plt = None

FILENAME_HELP = "ae2224I_measurement_data_subj<1-6>_C<1-6>.npz"
_DEFAULT_FS: float = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dir: Optional[str] = None
    save_dir: Optional[str] = None

    window_sizes: Tuple[int, ...] = (128, 256, 512, 1024)
    stride_fraction: float = 0.5

    input_combinations: Tuple[Tuple[str, ...], ...] = (
        ("e", "u"),
        ("e", "u", "de", "du"),
        ("e", "u", "de", "du", "d2e", "d2u"),
        ("e", "u", "e_vlo", "u_vlo", "e_lo", "u_lo"),
        ("e", "u", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
        ("e", "u", "eu", "u_de", "e_du"),
        ("e", "u", "eu", "u_de", "e_du", "de", "du"),
        ("e", "u", "de", "du", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
        ("e", "u", "eu", "u_de", "e_du", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
        (
            "e", "u", "eu", "u_de", "e_du", "de", "du",
            "e_vlo", "u_vlo", "e_lo", "u_lo",
            "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr",
            "e_logstd", "u_logstd",
        ),
    )

    batch_size: int = 256
    eval_batch_size: int = 2048
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    grad_clip_norm: float = 1.0

    # FIX 2: weight clamp range
    class_weight_min: float = 0.5
    class_weight_max: float = 2.0

    # FIX 1: fraction of training windows held out for threshold calibration
    threshold_cal_fraction: float = 0.30

    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 8
    val_check_interval: int = 2

    stage1_enabled: bool = True
    stage1_epochs: int = 8
    stage1_patience: int = 3
    stage1_top_k: int = 6
    global_tune_once_per_vehicle: bool = True

    random_state: int = 42
    min_run_length: int = 256
    default_fs: float = _DEFAULT_FS

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

    filename_pattern: str = (
        r"^ae2224I_measurement_data_subj(?P<subject>[1-6])_(?P<condition>C[1-6])\.npz$"
    )
    signal_keys: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {"e": ("e",), "u": ("u",), "time": ("t", "time")}
    )
    condition_to_label: Dict[str, int] = field(
        default_factory=lambda: {
            "C1": 0, "C2": 0, "C3": 0, "C4": 1, "C5": 1, "C6": 1,
        }
    )
    condition_to_vehicle: Dict[str, str] = field(
        default_factory=lambda: {
            "C1": "P", "C2": "V", "C3": "A",
            "C4": "P", "C5": "V", "C6": "A",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: per-fold threshold calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split_rows(rows: List[Dict[str, Any]],
                           cal_fraction: float,
                           rng: np.random.RandomState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split rows into (train_rows, calibration_rows) stratified by label.
    cal_fraction of each class goes to calibration.
    Operates at the window level so we keep as many training windows
    as possible for the actual model training.
    """
    by_class: DefaultDict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_class[int(row["y"])].append(i)

    cal_indices: List[int] = []
    for label, indices in by_class.items():
        indices_arr = np.asarray(indices)
        rng.shuffle(indices_arr)
        n_cal = max(1, int(round(len(indices_arr) * cal_fraction)))
        cal_indices.extend(indices_arr[:n_cal].tolist())

    cal_set = set(cal_indices)
    train_part = [rows[i] for i in range(len(rows)) if i not in cal_set]
    cal_part   = [rows[i] for i in cal_indices]
    return train_part, cal_part


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            grid_points: int = 161) -> float:
    """
    Grid-search threshold maximising Youden's J = sensitivity + specificity - 1.
    Falls back to 0.5 if either class is missing.
    """
    y_true = np.asarray(y_true, int); y_prob = np.asarray(y_prob, float)
    pos = int(np.sum(y_true == 1)); neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return 0.5
    best_j = -1.0; best_t = 0.5
    for t in np.linspace(0.05, 0.95, grid_points):
        yp = (y_prob >= t).astype(int)
        sens = float(np.sum((yp == 1) & (y_true == 1))) / pos
        spec = float(np.sum((yp == 0) & (y_true == 0))) / neg
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j; best_t = float(t)
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: capped class weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(rows: Sequence[Dict[str, Any]],
                           num_classes: int = 2,
                           w_min: float = 0.5,
                           w_max: float = 2.0,
                           device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Inverse-frequency class weights clamped to [w_min, w_max].
    Clamping prevents the over-correction that caused A to flip to
    a no-motion bias in v4.
    """
    counts = Counter(int(r["y"]) for r in rows)
    n_total = len(rows)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n_c = counts.get(c, 1)
        raw_w = n_total / (num_classes * max(1, n_c))
        weights[c] = float(np.clip(raw_w, w_min, w_max))
    if device is not None:
        weights = weights.to(device)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracker
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, total_trainings: int, heartbeat_seconds: float = 10.0):
        self.total_trainings = max(1, int(total_trainings))
        self.completed_trainings = 0
        self.start_time = time.time()
        self.current_label = ""; self.current_planned_epochs = 1
        self.current_epoch = 0; self.current_batch = 0
        self.current_total_batches = 1
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self.last_print_time = 0.0

    def start_training(self, label: str, planned_epochs: int) -> None:
        self.current_label = label
        self.current_planned_epochs = max(1, int(planned_epochs))
        self.current_epoch = 0; self.current_batch = 0
        self.current_total_batches = 1; self.last_print_time = 0.0
        self._print("START", nl=True)

    def start_epoch(self, epoch: int, total_batches: int) -> None:
        self.current_epoch = max(1, int(epoch)); self.current_batch = 0
        self.current_total_batches = max(1, int(total_batches))
        self._maybe("RUN", force=True)

    def update_batch(self, batch_idx: int, total_batches: int) -> None:
        self.current_batch = max(0, int(batch_idx))
        self.current_total_batches = max(1, int(total_batches))
        self._maybe("RUN")

    def finish_epoch(self, epoch: int) -> None:
        self.current_epoch = max(1, int(epoch))
        self.current_batch = self.current_total_batches
        self._print("RUN", nl=True)

    def finish_training(self, actual_epochs: int) -> None:
        self.current_epoch = max(0, int(actual_epochs))
        self.current_batch = self.current_total_batches
        self.completed_trainings += 1; self._print("DONE", nl=True)

    def _frac(self) -> float:
        bf = min(1.0, self.current_batch / max(1, self.current_total_batches))
        ef = 0.0
        if self.current_planned_epochs > 0 and self.current_epoch > 0:
            ef = ((self.current_epoch - 1) + bf) / self.current_planned_epochs
        return min(1.0, (self.completed_trainings + min(1.0, max(0.0, ef))) / self.total_trainings)

    @staticmethod
    def _fmt(s: float) -> str:
        if not np.isfinite(s) or s < 0: return "--:--:--"
        s = int(round(s)); h, r = divmod(s, 3600); m, ss = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{ss:02d}"

    def _msg(self, prefix: str) -> str:
        el = time.time() - self.start_time; frac = self._frac()
        eta = (el / frac - el) if frac > 1e-9 else float("inf")
        return (
            f"[{prefix}] {min(self.completed_trainings+1, self.total_trainings)}/"
            f"{self.total_trainings} | {frac*100:6.2f}% | "
            f"ep {self.current_epoch}/{self.current_planned_epochs} | "
            f"b {self.current_batch}/{self.current_total_batches} | "
            f"{self._fmt(el)} | ETA {self._fmt(eta)} | {self.current_label}"
        )

    def _print(self, prefix: str, nl: bool) -> None:
        msg = self._msg(prefix)
        if nl: print(msg, flush=True)
        else: print(msg, end="\r", flush=True)
        self.last_print_time = time.time()

    def _maybe(self, prefix: str, force: bool = False) -> None:
        if force or (time.time() - self.last_print_time >= self.heartbeat_seconds):
            self._print(prefix, nl=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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
    if device.type != "cuda":
        cfg.use_amp = False; cfg.pin_memory = False
        cfg.persistent_workers = False; cfg.num_workers = 0
        if cfg.eval_batch_size < cfg.batch_size: cfg.eval_batch_size = cfg.batch_size
        return cfg
    if cfg.num_workers < 0: cfg.num_workers = 0
    cpu_count = os.cpu_count() or 4
    if cfg.num_workers == 0: cfg.num_workers = min(8, max(2, cpu_count // 2))
    if running_in_notebook() and sys.platform.startswith("win"):
        cfg.num_workers = 0; cfg.persistent_workers = False; cfg.prefetch_factor = 2
    else:
        cfg.pin_memory = True
        if cfg.num_workers == 0: cfg.persistent_workers = False
    if cfg.eval_batch_size < cfg.batch_size: cfg.eval_batch_size = cfg.batch_size * 2
    return cfg

def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    if y_true.size == 0: return float("nan")
    return float(np.mean(y_true == y_pred))

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Sequence[int] = (0, 1)) -> np.ndarray:
    labels = list(labels); idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), int)
    for yt, yp in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        if yt in idx and yp in idx: cm[idx[yt], idx[yp]] += 1
    return cm

def classification_report_np(y_true: np.ndarray, y_pred: np.ndarray,
                              labels: Sequence[int] = (0, 1)) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    report: Dict[str, Dict[str, float]] = {}
    supports: List[int] = []; precs: List[float] = []; recs: List[float] = []; f1s: List[float] = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        sup = int(np.sum(y_true == label))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1   = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        report[str(label)] = {"precision": prec, "recall": rec, "f1-score": f1, "support": sup}
        supports.append(sup); precs.append(prec); recs.append(rec); f1s.append(f1)
    w = np.asarray(supports, float); ws = float(np.sum(w)) or 1.0; tot = int(np.sum(supports))
    report["accuracy"] = {"value": accuracy_score_np(y_true, y_pred)}
    report["macro avg"] = {"precision": float(np.mean(precs)), "recall": float(np.mean(recs)), "f1-score": float(np.mean(f1s)), "support": tot}
    report["weighted avg"] = {"precision": float(np.dot(precs, w)/ws), "recall": float(np.dot(recs, w)/ws), "f1-score": float(np.dot(f1s, w)/ws), "support": tot}
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32); mu = float(np.mean(x)); s = float(np.std(x))
    return (x - mu) / s if s > 1e-8 else x - mu

def mean_center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32); return x - float(np.mean(x))

def derivative(x: np.ndarray, time: Optional[np.ndarray] = None,
               fs: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if time is not None:
        t = np.asarray(time, np.float32).reshape(-1)
        return np.gradient(x, t).astype(np.float32)
    dt = 1.0 / fs if fs not in (None, 0) else 1.0
    return np.gradient(x, dt).astype(np.float32)

def _bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float,
              order: int = 3) -> np.ndarray:
    if not HAS_SCIPY_SIGNAL or len(x) < 20: return np.zeros_like(x)
    nyq = 0.5 * fs; lo = max(low_hz / nyq, 1e-4); hi = min(high_hz / nyq, 1 - 1e-4)
    if lo >= hi: return np.zeros_like(x)
    try:
        b, a = butter(order, [lo, hi], btype="band")
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        return np.zeros_like(x)

def normalize_run_signals(run: Dict[str, Any],
                           default_fs: float = _DEFAULT_FS) -> Dict[str, Any]:
    out = copy.deepcopy(run)
    raw_e = np.asarray(out["e"], np.float32)
    raw_u = np.asarray(out["u"], np.float32)
    time_vec = out.get("time"); fs = float(out.get("fs", default_fs)); n = len(raw_e)

    e_c = mean_center(raw_e); u_c = mean_center(raw_u)
    out["e"] = e_c; out["u"] = u_c

    de_raw = derivative(raw_e, time=time_vec, fs=fs)
    du_raw = derivative(raw_u, time=time_vec, fs=fs)
    out["de"] = safe_zscore(de_raw); out["du"] = safe_zscore(du_raw)
    out["d2e"] = safe_zscore(derivative(de_raw, time=time_vec, fs=fs))
    out["d2u"] = safe_zscore(derivative(du_raw, time=time_vec, fs=fs))

    std_e = float(np.std(e_c)) + 1e-8; std_u = float(np.std(u_c)) + 1e-8
    out["eu"]   = (e_c * u_c / (std_e * std_u)).astype(np.float32)
    out["u_de"] = safe_zscore(u_c * de_raw)
    out["e_du"] = safe_zscore(e_c * du_raw)

    e_vlo = _bandpass(raw_e, fs, 0.05, 0.5); u_vlo = _bandpass(raw_u, fs, 0.05, 0.5)
    e_lo  = _bandpass(raw_e, fs, 0.5,  3.0); u_lo  = _bandpass(raw_u, fs, 0.5,  3.0)
    out["e_vlo"] = safe_zscore(e_vlo); out["u_vlo"] = safe_zscore(u_vlo)
    out["e_lo"]  = safe_zscore(e_lo);  out["u_lo"]  = safe_zscore(u_lo)
    out["e_vlo_pwr"] = safe_zscore(e_vlo ** 2); out["u_vlo_pwr"] = safe_zscore(u_vlo ** 2)
    out["e_lo_pwr"]  = safe_zscore(e_lo  ** 2); out["u_lo_pwr"]  = safe_zscore(u_lo  ** 2)

    out["e_logstd"] = np.full(n, float(np.log1p(np.std(raw_e) + 1e-8)), np.float32)
    out["u_logstd"] = np.full(n, float(np.log1p(np.std(raw_u) + 1e-8)), np.float32)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _first_key(npz: Any, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for k in candidates:
        if k in npz.files: return k
    if required: raise KeyError(f"None of {tuple(candidates)} found in {npz.files}")
    return None

def _to_1d(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1: arr = np.asarray(arr.reshape(-1)[0])
    arr = np.asarray(arr, np.float32).squeeze()
    return arr.reshape(-1) if arr.ndim != 0 else arr.reshape(1)

def _split_runs(value: Any) -> List[np.ndarray]:
    arr = np.asarray(value)
    if arr.dtype == object: return [_to_1d(x) for x in arr.reshape(-1)]
    if arr.ndim <= 1: return [_to_1d(arr)]
    if arr.ndim == 2 and 1 in arr.shape:
        flat = arr.reshape(-1)
        return [_to_1d(x) for x in flat] if flat.dtype == object else [_to_1d(arr)]
    if arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]: return [_to_1d(arr[i]) for i in range(arr.shape[0])]
        return [_to_1d(arr[:, i]) for i in range(arr.shape[1])]
    return [_to_1d(arr)]

def _infer_fs(tv: Optional[np.ndarray]) -> Optional[float]:
    if tv is None or len(tv) < 2: return None
    dt = np.diff(np.asarray(tv, np.float32)); dt = dt[np.isfinite(dt)]
    if len(dt) == 0: return None
    md = float(np.median(dt))
    return None if abs(md) < 1e-12 else 1.0 / md

def resolve_data_dir(config: TrainConfig, script_dir: Path) -> Path:
    cands: List[Path] = []
    if config.data_dir not in (None, ""):
        p = Path(config.data_dir)
        cands += [p if p.is_absolute() else script_dir / p,
                  p if p.is_absolute() else Path.cwd() / p]
    cands += [
        script_dir / "data" / "python_data", script_dir / "python_data",
        Path.cwd() / "data" / "python_data",  Path.cwd() / "python_data",
        script_dir.parent / "data" / "python_data", script_dir.parent / "python_data",
    ]
    seen: set = set(); unique: List[Path] = []
    for c in cands:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen: seen.add(key); unique.append(c)
    for c in unique:
        if c.exists() and c.is_dir(): return c
    raise FileNotFoundError("Data dir not found. Checked:\n" + "\n".join(str(p) for p in unique))

def resolve_save_dir(config: TrainConfig, script_dir: Path) -> Path:
    if config.save_dir in (None, ""):
        return script_dir / "results_lstm_v5"
    p = Path(config.save_dir)
    return p if p.is_absolute() else script_dir / p

def load_runs_from_npz_dir(data_dir, config: TrainConfig,
                            min_run_length: int = 1) -> List[Dict[str, Any]]:
    data_path = Path(data_dir)
    if not data_path.exists(): raise FileNotFoundError(f"Not found: {data_path}")
    pattern = re.compile(config.filename_pattern, re.IGNORECASE)
    npz_files = [p for p in sorted(data_path.glob("*.npz")) if pattern.match(p.name)]
    if not npz_files: raise FileNotFoundError(f"No matching .npz in {data_path}")

    all_runs: List[Dict[str, Any]] = []
    for npz_file in npz_files:
        m = pattern.match(npz_file.name)
        if m is None: continue
        subject = f"subj{m.group('subject')}"
        condition = m.group("condition").upper()
        label = int(config.condition_to_label[condition])
        vehicle = str(config.condition_to_vehicle[condition])

        with np.load(npz_file, allow_pickle=True) as npz:
            e_key = _first_key(npz, config.signal_keys["e"])
            u_key = _first_key(npz, config.signal_keys["u"])
            t_key = _first_key(npz, config.signal_keys["time"], required=False)
            e_runs = _split_runs(npz[e_key]); u_runs = _split_runs(npz[u_key])
            if len(e_runs) != len(u_runs): raise ValueError(f"{npz_file.name}: e/u mismatch")
            if t_key is None:
                t_runs: List[Optional[np.ndarray]] = [None] * len(e_runs)
            else:
                raw_t = _split_runs(npz[t_key])
                if len(raw_t) == 1 and len(e_runs) > 1: t_runs = [raw_t[0]] * len(e_runs)
                elif len(raw_t) == len(e_runs): t_runs = list(raw_t)
                else: raise ValueError(f"{npz_file.name}: t count mismatch")

            for rep_idx, (e_r, u_r, t_r) in enumerate(zip(e_runs, u_runs, t_runs), 1):
                lengths = [len(e_r), len(u_r)]
                if t_r is not None: lengths.append(len(t_r))
                n = min(lengths)
                if n < min_run_length: continue
                run: Dict[str, Any] = {
                    "e": np.asarray(e_r[:n], np.float32),
                    "u": np.asarray(u_r[:n], np.float32),
                    "label": label, "vehicle_type": vehicle, "pilot_id": subject,
                    "condition_id": condition, "repetition_id": f"rep{rep_idx}",
                    "source_file": npz_file.name,
                    "run_id": f"{npz_file.stem}_rep{rep_idx}",
                }
                if t_r is not None:
                    run["time"] = np.asarray(t_r[:n], np.float32)
                    fs = _infer_fs(run["time"])
                    if fs is not None: run["fs"] = fs
                all_runs.append(normalize_run_signals(run, default_fs=config.default_fs))

    if not all_runs: raise ValueError("No valid runs loaded.")
    return all_runs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LazyWindowDataset(Dataset):
    def __init__(self, run_lookup, rows, input_vars):
        self.run_lookup = run_lookup; self.rows = list(rows)
        self.input_vars = tuple(input_vars)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]; run = self.run_lookup[row["run_id"]]
        s, e = int(row["start_idx"]), int(row["end_idx"])
        x = np.stack([run[var][s:e] for var in self.input_vars], axis=-1).astype(np.float32)
        meta = {k: row[k] for k in ("vehicle_type", "pilot_id", "repetition_id", "run_id", "source_file")}
        meta["start_idx"] = s; meta["end_idx"] = e
        return torch.from_numpy(x), torch.tensor(int(row["y"]), dtype=torch.long), meta


# ─────────────────────────────────────────────────────────────────────────────
# Architecture (unchanged from v4)
# ─────────────────────────────────────────────────────────────────────────────

class DilatedSlowPath(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, 7, padding=6,  dilation=2), nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 7, padding=12, dilation=4), nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 7, padding=24, dilation=8), nn.BatchNorm1d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class TemporalAttention(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(h, h//2), nn.Tanh(), nn.Linear(h//2, 1, bias=False))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(self.score(x), dim=1) * x).sum(dim=1)

class MultiScaleCNNBiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, num_classes: int = 2, cnn_channels: int = 64):
        super().__init__()
        c1 = cnn_channels // 3; c2 = cnn_channels // 3; c3 = cnn_channels - c1 - c2
        self.fast = nn.Sequential(nn.Conv1d(input_size, c1, 5,  padding=2),  nn.BatchNorm1d(c1), nn.GELU())
        self.mid  = nn.Sequential(nn.Conv1d(input_size, c2, 21, padding=10), nn.BatchNorm1d(c2), nn.GELU())
        self.slow = DilatedSlowPath(input_size, c3)
        self.fusion = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1), nn.BatchNorm1d(cnn_channels), nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1), nn.BatchNorm1d(cnn_channels), nn.GELU(),
        )
        eff_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers,
                            dropout=eff_drop, batch_first=True, bidirectional=True)
        h2 = hidden_size * 2
        self.attn = TemporalAttention(h2)
        self.res_proj = nn.Linear(cnn_channels, h2)
        self.head = nn.Sequential(
            nn.LayerNorm(h2), nn.Linear(h2, hidden_size),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xT = x.permute(0, 2, 1)
        cnn = self.fusion(torch.cat([self.fast(xT), self.mid(xT), self.slow(xT)], dim=1))
        seq = cnn.permute(0, 2, 1)
        out, _ = self.lstm(seq)
        return self.head(self.attn(out) + self.res_proj(seq.mean(dim=1)))

def build_model(config: TrainConfig, input_size: int, device: torch.device) -> nn.Module:
    model = MultiScaleCNNBiLSTMClassifier(
        input_size=input_size, hidden_size=config.hidden_size,
        num_layers=config.num_layers, dropout=config.dropout, cnn_channels=64,
    ).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        try: model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception: pass
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)

def make_loader(rows, run_lookup, input_vars, batch_size, shuffle, config, device):
    ds = LazyWindowDataset(run_lookup, rows, input_vars)
    kw: Dict[str, Any] = {
        "batch_size": batch_size, "shuffle": shuffle,
        "num_workers": int(config.num_workers), "collate_fn": collate_fn,
        "pin_memory": bool(config.pin_memory and device.type == "cuda"), "drop_last": False,
    }
    if config.num_workers > 0:
        kw["persistent_workers"] = bool(config.persistent_workers)
        kw["prefetch_factor"] = int(config.prefetch_factor)
    return DataLoader(ds, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# CV helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_unique_pilot_ids(runs: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(r["pilot_id"]) for r in runs})

def build_candidate_cache(vehicle_runs, config):
    run_lookup = {str(r["run_id"]): r for r in vehicle_runs}
    cache: Dict[Tuple, Dict[str, Any]] = {}
    for input_vars in config.input_combinations:
        for window_size in config.window_sizes:
            stride = max(1, int(round(window_size * config.stride_fraction)))
            rbp: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
            total = 0
            for run in vehicle_runs:
                rid = str(run["run_id"]); pid = str(run["pilot_id"]); n = len(run["e"])
                if n < window_size: continue
                for s in range(0, n - window_size + 1, stride):
                    rbp[pid].append({
                        "run_id": rid, "pilot_id": pid,
                        "vehicle_type": str(run["vehicle_type"]),
                        "repetition_id": str(run["repetition_id"]),
                        "source_file": str(run["source_file"]),
                        "start_idx": int(s), "end_idx": int(s + window_size),
                        "y": int(run["label"]),
                    })
                    total += 1
            if total > 0:
                cache[(tuple(input_vars), int(window_size))] = {
                    "input_vars": tuple(input_vars), "window_size": int(window_size),
                    "rows_by_pilot": dict(rbp), "total_windows": int(total),
                }
    return run_lookup, cache

def collect_rows_for_pilots(ce, pilot_ids):
    out: List[Dict[str, Any]] = []
    for pid in pilot_ids: out.extend(ce["rows_by_pilot"].get(str(pid), []))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _should_val(epoch, max_epochs, vci):
    return epoch == 1 or epoch == max_epochs or epoch % max(1, int(vci)) == 0

def _make_criterion(train_rows, config, device):
    w = compute_class_weights(train_rows, num_classes=2,
                               w_min=config.class_weight_min,
                               w_max=config.class_weight_max,
                               device=device)
    return nn.CrossEntropyLoss(weight=w)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp,
                    grad_clip_norm: float = 1.0, tracker=None, batch_update_interval: int = 50):
    model.train(); total_loss = 0.0; total_n = 0; nb = len(loader)
    for bi, (x, y, _) in enumerate(loader, 1):
        if device.type == "cuda":
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        else:
            x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        bs = int(x.size(0)); total_loss += float(loss.item()) * bs; total_n += bs
        if tracker and (bi == 1 or bi == nb or bi % max(1, batch_update_interval) == 0):
            tracker.update_batch(bi, nb)
    return total_loss / max(1, total_n)

@torch.inference_mode()
def evaluate_accuracy_only(model, loader, device, use_amp):
    model.eval(); correct = 0; total = 0
    for x, y, _ in loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
        else:
            x = x.to(device); y = y.to(device); logits = model(x)
        correct += int((torch.argmax(logits, 1) == y).sum()); total += int(y.numel())
    return float(correct / total) if total > 0 else float("nan")

@torch.inference_mode()
def predict_loader(model, loader, device, use_amp):
    model.eval()
    y_true: List[int] = []; y_pred: List[int] = []; y_prob: List[float] = []; metas: List[Dict] = []
    for x, y, meta in loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
        else:
            x = x.to(device); logits = model(x)
        probs = torch.softmax(logits, 1).detach().cpu().numpy()
        y_true.extend(y.numpy().tolist()); y_pred.extend(np.argmax(probs, 1).tolist())
        y_prob.extend(probs[:, 1].tolist()); metas.extend(meta)
    return (np.asarray(y_true, int), np.asarray(y_pred, int),
            np.asarray(y_prob, float), metas)

def fit_lstm(train_rows, val_rows, run_lookup, input_vars, config, device,
             tracker=None, training_label="", max_epochs_override=None,
             patience_override=None, val_check_interval_override=None):
    """Standard training with early stopping. Returns (model, history, best_epoch, best_val_acc)."""
    model = build_model(config, input_size=len(input_vars), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    max_epochs = int(max_epochs_override or config.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, min(10, max_epochs // 3)), eta_min=1e-5)
    criterion = _make_criterion(train_rows, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if device.type == "cuda" else None
    train_loader = make_loader(train_rows, run_lookup, input_vars, config.batch_size, True, config, device)
    val_loader   = make_loader(val_rows,   run_lookup, input_vars, config.eval_batch_size, False, config, device)
    patience = int(patience_override or config.patience)
    vci = int(val_check_interval_override or config.val_check_interval)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -float("inf"); best_epoch = 1
    history: List[Dict[str, Any]] = []; wait = 0

    if tracker: tracker.start_training(training_label or f"train={len(train_rows)}", max_epochs)

    for epoch in range(1, max_epochs + 1):
        if tracker: tracker.start_epoch(epoch, len(train_loader))
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device,
                              scaler, config.use_amp, grad_clip_norm=config.grad_clip_norm,
                              tracker=tracker, batch_update_interval=config.batch_update_interval)
        scheduler.step(epoch)
        if _should_val(epoch, max_epochs, vci):
            va = evaluate_accuracy_only(model, val_loader, device, config.use_amp)
            history.append({"epoch": epoch, "train_loss": tl, "val_accuracy": va})
            if va > best_val_acc:
                best_val_acc = va; best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict()); wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if tracker: tracker.finish_epoch(epoch); break
        else:
            history.append({"epoch": epoch, "train_loss": tl, "val_accuracy": None})
        if tracker: tracker.finish_epoch(epoch)

    model.load_state_dict(best_state)
    if tracker: tracker.finish_training(actual_epochs=history[-1]["epoch"] if history else 0)
    return model, history, best_epoch, best_val_acc


def fit_lstm_fixed_epochs(train_rows, run_lookup, input_vars, config, device,
                           num_epochs, tracker=None, training_label=""):
    """Train for a fixed number of epochs; return model."""
    model = build_model(config, input_size=len(input_vars), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    planned = max(1, int(num_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, min(10, planned // 3)), eta_min=1e-5)
    criterion = _make_criterion(train_rows, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if device.type == "cuda" else None
    train_loader = make_loader(train_rows, run_lookup, input_vars, config.batch_size, True, config, device)
    history: List[Dict[str, Any]] = []

    if tracker: tracker.start_training(training_label or f"final={len(train_rows)}", planned)
    for epoch in range(1, planned + 1):
        if tracker: tracker.start_epoch(epoch, len(train_loader))
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device,
                              scaler, config.use_amp, grad_clip_norm=config.grad_clip_norm,
                              tracker=tracker, batch_update_interval=config.batch_update_interval)
        scheduler.step(epoch)
        history.append({"epoch": epoch, "train_loss": tl})
        if tracker: tracker.finish_epoch(epoch)
    if tracker: tracker.finish_training(actual_epochs=planned)
    return model, history


def calibrate_threshold_on_training_data(
    model: nn.Module,
    train_rows: List[Dict[str, Any]],
    run_lookup: Dict[str, Any],
    input_vars: Sequence[str],
    config: TrainConfig,
    device: torch.device,
    rng: np.random.RandomState,
) -> float:
    """
    FIX 1: Calibrate the decision threshold for this specific outer fold.

    Split a stratified 30% of training windows aside (never seen during
    training), run the model over them, and find the threshold that
    maximises Youden's J.  Using training-data calibration is valid because
    these windows are only used to set the threshold, not to select
    architecture or train weights.

    Returns the calibrated threshold (falls back to 0.5 if calibration
    fails due to too few examples).
    """
    # Stratified split: 30% for calibration, rest discarded (not re-trained)
    _, cal_rows = stratified_split_rows(train_rows, config.threshold_cal_fraction, rng)

    labels_present = {int(r["y"]) for r in cal_rows}
    if len(labels_present) < 2 or len(cal_rows) < 10:
        return 0.5

    cal_loader = make_loader(cal_rows, run_lookup, input_vars,
                             config.eval_batch_size, False, config, device)
    y_true_cal, _, y_prob_cal, _ = predict_loader(model, cal_loader, device, config.use_amp)

    # Aggregate to run level before calibrating — this matches how the
    # final predictions will be evaluated (avoids window-level majority vote
    # dominating the threshold)
    run_grouped: DefaultDict[str, List] = defaultdict(list)
    run_labels:  Dict[str, int] = {}
    for row, yt, yp in zip(cal_rows, y_true_cal, y_prob_cal):
        rid = str(row["run_id"])
        run_grouped[rid].append(float(yp))
        run_labels[rid] = int(yt)

    run_y_true = np.asarray([run_labels[rid] for rid in run_grouped], int)
    run_y_prob = np.asarray([float(np.mean(scores)) for scores in run_grouped.values()], float)

    return find_optimal_threshold(run_y_true, run_y_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction & metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_rows(eval_result, threshold: float = 0.5):
    rows = []
    for yt, _, prob, meta in zip(eval_result["y_true"], eval_result["y_pred"],
                                  eval_result["y_prob"], eval_result["metas"]):
        rows.append({
            "true_label": int(yt), "pred_label": int(prob >= threshold),
            "motion_on_confidence": float(prob),
            "vehicle_type": meta["vehicle_type"], "pilot_id": meta["pilot_id"],
            "repetition_id": meta["repetition_id"], "run_id": meta["run_id"],
            "source_file": meta["source_file"],
            "start_idx": int(meta["start_idx"]), "end_idx": int(meta["end_idx"]),
        })
    return rows

def aggregate_run_predictions(pred_rows, threshold: float = 0.5):
    grouped: DefaultDict = defaultdict(list)
    for row in pred_rows:
        key = (str(row["vehicle_type"]), str(row["pilot_id"]), str(row["repetition_id"]),
               str(row["run_id"]), int(row["true_label"]))
        grouped[key].append(float(row["motion_on_confidence"]))
    out = []
    for (vt, pid, rep, rid, tl), scores in grouped.items():
        mc = float(np.mean(scores))
        out.append({"vehicle_type": vt, "pilot_id": pid, "repetition_id": rep, "run_id": rid,
                    "true_label": tl, "pred_label": int(mc >= threshold),
                    "motion_on_confidence": mc, "num_windows": int(len(scores))})
    return sorted(out, key=lambda r: (r["vehicle_type"], r["pilot_id"], r["repetition_id"], r["run_id"]))

def compute_metrics_from_prediction_rows(pred_rows):
    y_true = np.asarray([int(r["true_label"]) for r in pred_rows], int)
    y_pred = np.asarray([int(r["pred_label"]) for r in pred_rows], int)
    y_prob = np.asarray([float(r["motion_on_confidence"]) for r in pred_rows], float)
    return {
        "accuracy": accuracy_score_np(y_true, y_pred),
        "confusion_matrix": confusion_matrix_np(y_true, y_pred, labels=(0, 1)),
        "classification_report": classification_report_np(y_true, y_pred, labels=(0, 1)),
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }

def run_basic_statistics(run_pred_rows):
    results: Dict[str, Any] = {"aggregated_run_predictions": list(run_pred_rows)}
    if not HAS_SCIPY: return results
    off = np.asarray([r["motion_on_confidence"] for r in run_pred_rows if int(r["true_label"]) == 0])
    on  = np.asarray([r["motion_on_confidence"] for r in run_pred_rows if int(r["true_label"]) == 1])
    if len(off) >= 2 and len(on) >= 2:
        ts, pv = scipy_stats.ttest_ind(off, on, equal_var=False)  # type: ignore[union-attr]
        results["t_test_confidence_off_vs_on"] = {
            "n_off_runs": int(len(off)), "n_on_runs": int(len(on)),
            "mean_off": float(np.mean(off)), "mean_on": float(np.mean(on)),
            "t_statistic": float(ts), "p_value": float(pv),
        }
    return results

def compute_bias_matrix(run_pred_rows: Sequence[Dict[str, Any]],
                         threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray([int(r["true_label"]) for r in run_pred_rows])
    y_prob = np.asarray([float(r["motion_on_confidence"]) for r in run_pred_rows])
    y_pred = (y_prob >= threshold).astype(int)
    pos = np.sum(y_true == 1); neg = np.sum(y_true == 0)
    tp = float(np.sum((y_pred == 1) & (y_true == 1))); tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0))); fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / pos if pos > 0 else 0.0; spec = tn / neg if neg > 0 else 0.0
    fpr  = fp / neg if neg > 0 else 0.0; fnr  = fn / pos if pos > 0 else 0.0
    bias = fpr - fnr
    return {
        "threshold": float(threshold),
        "n_no_motion_runs": int(neg), "n_motion_runs": int(pos),
        "accuracy_no_motion": float(spec), "accuracy_motion": float(sens),
        "false_positive_rate": float(fpr), "false_negative_rate": float(fnr),
        "bias": float(bias),
        "bias_direction": "motion" if bias > 0.05 else ("no_motion" if bias < -0.05 else "balanced"),
    }

def print_bias_matrix(bm: Dict[str, Any], vehicle_type: str) -> None:
    print(f"\n{'─'*56}")
    print(f"  Bias matrix — {vehicle_type}   (threshold={bm['threshold']:.3f})")
    print(f"{'─'*56}")
    print(f"  {'Class':<22}  {'n':>4}  {'Accuracy':>9}  {'Error rate':>10}")
    print(f"  {'No motion':<22}  {bm['n_no_motion_runs']:>4}  {bm['accuracy_no_motion']:>9.3f}  {bm['false_positive_rate']:>10.3f}")
    print(f"  {'Motion':<22}  {bm['n_motion_runs']:>4}  {bm['accuracy_motion']:>9.3f}  {bm['false_negative_rate']:>10.3f}")
    print(f"  Bias (FPR − FNR) : {bm['bias']:+.3f}  → {bm['bias_direction']}")
    print(f"{'─'*56}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment orchestration
# ─────────────────────────────────────────────────────────────────────────────

def estimate_total_trainings(runs_by_vehicle, config):
    total = 0; nc = len(config.window_sizes) * len(config.input_combinations)
    s2c = min(nc, max(1, int(config.stage1_top_k))) if config.stage1_enabled else nc
    for vruns in runs_by_vehicle.values():
        labels = {int(r["label"]) for r in vruns}; pids = get_unique_pilot_ids(vruns)
        if len(labels) < 2 or len(pids) < 3: continue
        np_ = len(pids)
        if config.global_tune_once_per_vehicle:
            total += np_ * (nc + s2c if config.stage1_enabled else nc) + np_
        else:
            inner = np_ - 1
            total += np_ * (inner * (nc + s2c if config.stage1_enabled else nc) + 1)
    return total

def run_inner_cv_hyperparameter_search(train_pilot_ids, vehicle_type, config, device,
                                        run_lookup, candidate_cache, tracker=None):
    """Inner CV for hyperparameter selection ONLY (no threshold calibration here)."""
    train_pilot_ids = list(train_pilot_ids)
    if len(train_pilot_ids) < 2: raise ValueError("Need ≥2 pilots for inner CV.")
    candidate_keys = list(candidate_cache.keys())
    stage1_rows: List[Dict[str, Any]] = []

    if config.stage1_enabled:
        for ivars, wsize in candidate_keys:
            ce = candidate_cache[(tuple(ivars), int(wsize))]
            accs: List[float] = []; epochs: List[int] = []; failed = False
            for vp in train_pilot_ids:
                tr = collect_rows_for_pilots(ce, [p for p in train_pilot_ids if p != vp])
                vr = collect_rows_for_pilots(ce, [vp])
                if not tr or not vr or len({int(r["y"]) for r in tr}) < 2:
                    failed = True; break
                _, _, be, bva = fit_lstm(tr, vr, run_lookup, ivars, config, device,
                    tracker=tracker,
                    training_label=f"{vehicle_type}|s1|{vp}|{','.join(ivars)}|w{wsize}",
                    max_epochs_override=config.stage1_epochs,
                    patience_override=config.stage1_patience,
                    val_check_interval_override=max(2, config.val_check_interval))
                accs.append(float(bva)); epochs.append(int(be))
            if not failed and accs:
                stage1_rows.append({
                    "vehicle_type": vehicle_type, "input_vars": ",".join(ivars),
                    "window_size": int(wsize),
                    "mean_stage1_val_accuracy": float(np.mean(accs)),
                    "std_stage1_val_accuracy":  float(np.std(accs, ddof=0)),
                    "mean_stage1_best_epoch":   float(np.mean(epochs)),
                })
        if not stage1_rows: raise ValueError("Stage-1 found no valid settings.")
        stage1_rows.sort(key=lambda r: (-r["mean_stage1_val_accuracy"], r["std_stage1_val_accuracy"], r["window_size"]))
        selected = stage1_rows[:min(len(stage1_rows), max(1, int(config.stage1_top_k)))]
        selected_keys = [(tuple(r["input_vars"].split(",")), int(r["window_size"])) for r in selected]
    else:
        selected_keys = candidate_keys

    summary_rows: List[Dict[str, Any]] = []; fold_rows: List[Dict[str, Any]] = []
    for ivars, wsize in selected_keys:
        ce = candidate_cache.get((tuple(ivars), int(wsize)))
        if ce is None: continue
        cand_folds: List[Dict[str, Any]] = []; failed = False
        for vp in train_pilot_ids:
            tr = collect_rows_for_pilots(ce, [p for p in train_pilot_ids if p != vp])
            vr = collect_rows_for_pilots(ce, [vp])
            if not tr or not vr or len({int(r["y"]) for r in tr}) < 2:
                failed = True; break
            _, _, be, bva = fit_lstm(tr, vr, run_lookup, ivars, config, device,
                tracker=tracker,
                training_label=f"{vehicle_type}|inner|{vp}|{','.join(ivars)}|w{wsize}")
            cand_folds.append({
                "vehicle_type": vehicle_type, "validation_pilot": vp,
                "input_vars": ",".join(ivars), "window_size": int(wsize),
                "best_epoch": int(be), "best_val_accuracy": float(bva),
            })
        if failed or not cand_folds: continue
        fold_rows.extend(cand_folds)
        accs = [r["best_val_accuracy"] for r in cand_folds]
        epcs = [r["best_epoch"] for r in cand_folds]
        summary_rows.append({
            "vehicle_type": vehicle_type, "input_vars": ",".join(ivars),
            "window_size": int(wsize),
            "mean_inner_val_accuracy": float(np.mean(accs)),
            "std_inner_val_accuracy":  float(np.std(accs, ddof=0)),
            "mean_best_epoch":         float(np.mean(epcs)),
            "num_inner_folds":         int(len(cand_folds)),
            "total_windows_candidate": int(ce["total_windows"]),
        })
    if not summary_rows: raise ValueError("Inner CV found no valid settings.")
    summary_rows.sort(key=lambda r: (-r["mean_inner_val_accuracy"], r["std_inner_val_accuracy"], r["window_size"]))
    return summary_rows[0], summary_rows, fold_rows


def run_vehicle_experiment(runs_for_vehicle, vehicle_type, config, device, tracker=None):
    pilot_ids = get_unique_pilot_ids(runs_for_vehicle)
    if len(pilot_ids) < 3: raise ValueError(f"{vehicle_type}: need ≥3 pilots.")

    # Use a fixed RNG seeded from config for reproducible threshold calibration
    cal_rng = np.random.RandomState(config.random_state + hash(vehicle_type) % 10000)

    outer_fold_rows: List[Dict[str, Any]] = []
    combined_window_rows: List[Dict[str, Any]] = []
    combined_run_rows: List[Dict[str, Any]] = []

    run_lookup_all, candidate_cache_all = build_candidate_cache(runs_for_vehicle, config)

    global_best_ivars: Optional[Tuple[str, ...]] = None
    global_best_wsize: Optional[int] = None
    global_best_epochs: Optional[int] = None

    if config.global_tune_once_per_vehicle:
        print(f"\nVehicle {vehicle_type}: shared hyperparameter search")
        filtered = {k: v for k, v in candidate_cache_all.items()
                    if all(v["rows_by_pilot"].get(p, []) for p in pilot_ids)}
        if not filtered: raise ValueError(f"{vehicle_type}: no candidate had data for every pilot.")
        best_row, all_summary, _ = run_inner_cv_hyperparameter_search(
            pilot_ids, vehicle_type, config, device, run_lookup_all, filtered, tracker)
        global_best_ivars  = tuple(str(best_row["input_vars"]).split(","))
        global_best_wsize  = int(best_row["window_size"])
        global_best_epochs = max(1, int(round(float(best_row["mean_best_epoch"]))))
        print(f"  → window={global_best_wsize}, vars={','.join(global_best_ivars)}, epochs={global_best_epochs}")
        for row in all_summary[:3]:
            print(f"     candidate: acc={row['mean_inner_val_accuracy']:.3f}±{row['std_inner_val_accuracy']:.3f}"
                  f"  w={row['window_size']}  vars={row['input_vars']}")

    for test_pilot in pilot_ids:
        outer_train_pilots = [p for p in pilot_ids if p != test_pilot]
        outer_train_runs   = [r for r in runs_for_vehicle if str(r["pilot_id"]) != str(test_pilot)]
        if len({int(r["label"]) for r in outer_train_runs}) < 2:
            print(f"Skipping {vehicle_type}/{test_pilot}: single class"); continue
        print(f"\n{vehicle_type}: outer fold  test={test_pilot}")

        if config.global_tune_once_per_vehicle:
            best_ivars = global_best_ivars; best_wsize = global_best_wsize
            best_epochs = global_best_epochs
        else:
            filtered = {k: v for k, v in candidate_cache_all.items()
                        if all(v["rows_by_pilot"].get(p, []) for p in outer_train_pilots + [test_pilot])}
            best_row, _, _ = run_inner_cv_hyperparameter_search(
                outer_train_pilots, vehicle_type, config, device, run_lookup_all, filtered, tracker)
            best_ivars  = tuple(str(best_row["input_vars"]).split(","))
            best_wsize  = int(best_row["window_size"])
            best_epochs = max(1, int(round(float(best_row["mean_best_epoch"]))))

        ce = candidate_cache_all[(best_ivars, best_wsize)]
        train_rows = collect_rows_for_pilots(ce, outer_train_pilots)
        test_rows  = collect_rows_for_pilots(ce, [test_pilot])

        # Log class balance
        counts = Counter(int(r["y"]) for r in train_rows)
        w = compute_class_weights(train_rows, num_classes=2,
                                   w_min=config.class_weight_min,
                                   w_max=config.class_weight_max)
        print(f"  training windows: class0={counts[0]}, class1={counts[1]}, "
              f"weights=[{w[0]:.3f}, {w[1]:.3f}]")

        final_model, _ = fit_lstm_fixed_epochs(
            train_rows, run_lookup_all, best_ivars, config, device,
            num_epochs=best_epochs, tracker=tracker,
            training_label=f"{vehicle_type}|final|test={test_pilot}|w={best_wsize}",
        )

        # FIX 1: calibrate threshold on training data for THIS outer fold
        opt_threshold = calibrate_threshold_on_training_data(
            final_model, train_rows, run_lookup_all, best_ivars, config, device, cal_rng)
        print(f"  calibrated threshold = {opt_threshold:.3f}")

        test_loader = make_loader(test_rows, run_lookup_all, best_ivars,
                                  config.eval_batch_size, False, config, device)
        y_true, y_pred, y_prob, metas = predict_loader(final_model, test_loader, device, config.use_amp)
        window_rows = build_prediction_rows(
            {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "metas": metas},
            threshold=opt_threshold)
        for row in window_rows:
            row.update({"outer_test_pilot": test_pilot,
                        "selected_input_vars": ",".join(best_ivars),
                        "selected_window_size": best_wsize,
                        "selected_num_epochs": best_epochs,
                        "opt_threshold": opt_threshold})

        run_rows = aggregate_run_predictions(window_rows, threshold=opt_threshold)
        for row in run_rows:
            row.update({"outer_test_pilot": test_pilot,
                        "selected_input_vars": ",".join(best_ivars),
                        "selected_window_size": best_wsize,
                        "selected_num_epochs": best_epochs,
                        "opt_threshold": opt_threshold})

        run_metrics = compute_metrics_from_prediction_rows(run_rows)
        outer_fold_rows.append({
            "vehicle_type": vehicle_type, "outer_test_pilot": test_pilot,
            "selected_input_vars": ",".join(best_ivars),
            "selected_window_size": best_wsize, "selected_num_epochs": best_epochs,
            "opt_threshold": opt_threshold,
            "num_test_runs": int(len(run_rows)),
            "run_level_accuracy": float(run_metrics["accuracy"]),
        })
        combined_window_rows.extend(window_rows); combined_run_rows.extend(run_rows)

    if not combined_run_rows: raise ValueError(f"No valid outer folds for {vehicle_type}.")
    mean_threshold = float(np.mean([r.get("opt_threshold", 0.5) for r in outer_fold_rows]))
    test_result = compute_metrics_from_prediction_rows(combined_run_rows)
    statistics  = run_basic_statistics(combined_run_rows)
    bias_matrix = compute_bias_matrix(combined_run_rows, threshold=mean_threshold)

    return {
        "vehicle_type": vehicle_type, "outer_fold_rows": outer_fold_rows,
        "window_prediction_rows": combined_window_rows, "run_prediction_rows": combined_run_rows,
        "test_result": test_result, "statistics": statistics,
        "bias_matrix": bias_matrix, "mean_threshold": mean_threshold,
        "num_outer_folds": int(len(outer_fold_rows)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O & plotting
# ─────────────────────────────────────────────────────────────────────────────

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
        w = csv.DictWriter(f, fieldnames=list(fieldnames)); w.writeheader()
        for row in rows: w.writerow(row)

def _save_plot_cm(cm, path, title):
    if plt is None: return
    fig, ax = plt.subplots(figsize=(4.5, 4)); im = ax.imshow(cm)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: No motion","Pred: Motion"])
    ax.set_yticklabels(["Actual: No motion","Actual: Motion"]); ax.set_title(title)
    for i in range(2):
        for j in range(2): ax.text(j, i, str(int(cm[i,j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)

def _save_plot_bias(bm: Dict[str, Any], path: Path, vehicle_type: str) -> None:
    if plt is None: return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    bars = ax.bar(["No motion","Motion"], [bm["accuracy_no_motion"],bm["accuracy_motion"]],
                  color=["#3a86ff","#ff006e"], width=0.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class accuracy — {vehicle_type}")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8)
    for bar, v in zip(bars, [bm["accuracy_no_motion"],bm["accuracy_motion"]]):
        ax.text(bar.get_x()+bar.get_width()/2, min(0.97,v+0.02), f"{v:.2f}", ha="center", fontsize=11)
    ax2 = axes[1]
    bars2 = ax2.bar(["FPR\n(no-motion→motion)","FNR\n(motion→no-motion)"],
                    [bm["false_positive_rate"],bm["false_negative_rate"]],
                    color=["#ffbe0b","#fb5607"], width=0.5)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Error rate")
    ax2.set_title(f"Bias: {bm['bias']:+.3f} ({bm['bias_direction']})\nThreshold={bm['threshold']:.3f}")
    for bar, v in zip(bars2, [bm["false_positive_rate"],bm["false_negative_rate"]]):
        ax2.text(bar.get_x()+bar.get_width()/2, min(0.97,v+0.02), f"{v:.2f}", ha="center", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def _save_plot_pilot_accuracy(outer_fold_rows, path, title):
    if plt is None or not outer_fold_rows: return
    pilots = [str(r["outer_test_pilot"]) for r in outer_fold_rows]
    vals = [float(r["run_level_accuracy"]) for r in outer_fold_rows]
    thresholds = [float(r.get("opt_threshold", 0.5)) for r in outer_fold_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pilots, vals); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_title(title)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8)
    for i, (v, t) in enumerate(zip(vals, thresholds)):
        ax.text(i, min(0.97, v+0.02), f"{v:.2f}\n(t={t:.2f})", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def _save_plot_overall(summary_rows, path):
    if plt is None or not summary_rows: return
    vehicles = [str(r["vehicle_type"]) for r in summary_rows]
    vals = [float(r["run_level_accuracy"]) for r in summary_rows]
    colors = {"P":"#3a86ff","V":"#ff006e","A":"#8338ec"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(vehicles, vals, color=[colors.get(v,"#aaa") for v in vehicles])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Run-level accuracy")
    ax.set_title("Overall accuracy by vehicle type (v5)")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, label="80% target"); ax.legend(fontsize=9)
    for bar, v in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, min(0.97,v+0.02), f"{v:.2f}", ha="center", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def save_vehicle_results(result, save_dir):
    vd = Path(save_dir) / str(result["vehicle_type"]); vd.mkdir(parents=True, exist_ok=True)
    write_csv(vd / "pilot_results.csv", result["outer_fold_rows"])
    write_csv(vd / "run_predictions.csv", result["run_prediction_rows"])
    cm = result["test_result"]["confusion_matrix"]
    write_csv(vd / "confusion_matrix.csv",
              [{"actual":"no_motion","pred_no_motion":int(cm[0,0]),"pred_motion":int(cm[0,1])},
               {"actual":"motion",   "pred_no_motion":int(cm[1,0]),"pred_motion":int(cm[1,1])}],
              fieldnames=["actual","pred_no_motion","pred_motion"])
    _save_plot_cm(cm, vd/"confusion_matrix.png", f"Vehicle {result['vehicle_type']}")
    _save_plot_pilot_accuracy(result["outer_fold_rows"], vd/"pilot_accuracy.png",
                              f"{result['vehicle_type']} — accuracy by held-out pilot")
    _save_plot_bias(result["bias_matrix"], vd/"bias_matrix.png", str(result["vehicle_type"]))
    bm = result["bias_matrix"]
    summary = {
        "vehicle_type": result["vehicle_type"], "outer_folds": result["num_outer_folds"],
        "mean_threshold": result["mean_threshold"],
        "run_level_accuracy": result["test_result"]["accuracy"],
        "confusion_matrix": cm.tolist(), "bias_matrix": bm,
        "statistics": {k: v for k, v in result["statistics"].items() if k != "aggregated_run_predictions"},
    }
    (vd/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (vd/"summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Vehicle type            : {result['vehicle_type']}\n")
        f.write(f"Outer folds             : {result['num_outer_folds']}\n")
        f.write(f"Mean calibrated threshold: {result['mean_threshold']:.3f}\n")
        f.write(f"Overall run accuracy    : {result['test_result']['accuracy']:.3f}\n")
        f.write("Confusion matrix:\n")
        f.write(f"  No motion → no motion: {int(cm[0,0])}\n")
        f.write(f"  No motion → motion   : {int(cm[0,1])}\n")
        f.write(f"  Motion    → no motion: {int(cm[1,0])}\n")
        f.write(f"  Motion    → motion   : {int(cm[1,1])}\n")
        f.write(f"Bias matrix:\n")
        f.write(f"  Accuracy (no motion) : {bm['accuracy_no_motion']:.3f}\n")
        f.write(f"  Accuracy (motion)    : {bm['accuracy_motion']:.3f}\n")
        f.write(f"  False positive rate  : {bm['false_positive_rate']:.3f}\n")
        f.write(f"  False negative rate  : {bm['false_negative_rate']:.3f}\n")
        f.write(f"  Bias (FPR−FNR)       : {bm['bias']:+.3f}  ({bm['bias_direction']})\n")
        if "t_test_confidence_off_vs_on" in result["statistics"]:
            tt = result["statistics"]["t_test_confidence_off_vs_on"]
            f.write(f"\nConfidence: no-motion={tt['mean_off']:.3f}, motion={tt['mean_on']:.3f}, p={tt['p_value']:.4g}\n")
    return {"vehicle_type": result["vehicle_type"],
            "run_level_accuracy": float(result["test_result"]["accuracy"]),
            "outer_folds": int(result["num_outer_folds"]),
            "mean_threshold": float(result["mean_threshold"])}

def print_dataset_summary(runs):
    print("=" * 72)
    print(f"Dataset  |  runs={len(runs)}  |  scipy.signal={HAS_SCIPY_SIGNAL}")
    by_v = Counter(str(r["vehicle_type"]) for r in runs)
    by_l = Counter(int(r["label"]) for r in runs)
    print(f"Pilots: {', '.join(sorted({str(r['pilot_id']) for r in runs}))}")
    print(f"By vehicle: {dict(by_v)}  |  No-motion: {by_l.get(0,0)}  Motion: {by_l.get(1,0)}")
    if not HAS_SCIPY_SIGNAL:
        print("⚠  scipy not found — bandpass channels will be zero.  pip install scipy")
    print("=" * 72)

def print_summary(result):
    cm = result["test_result"]["confusion_matrix"]
    print("=" * 72)
    print(f"Vehicle: {result['vehicle_type']}  |  folds={result['num_outer_folds']}  "
          f"|  accuracy={result['test_result']['accuracy']:.3f}  "
          f"|  mean_threshold={result['mean_threshold']:.3f}")
    print(f"Confusion matrix:  TN={int(cm[0,0])} FP={int(cm[0,1])} FN={int(cm[1,0])} TP={int(cm[1,1])}")
    print_bias_matrix(result["bias_matrix"], str(result["vehicle_type"]))
    if result["outer_fold_rows"]:
        print(f"  {'Pilot':<12}{'Acc':>8}{'Window':>8}{'Thresh':>8}  Inputs")
        for row in result["outer_fold_rows"]:
            print(f"  {str(row['outer_test_pilot']):<12}"
                  f"{float(row['run_level_accuracy']):>8.3f}"
                  f"{int(row['selected_window_size']):>8}"
                  f"{float(row.get('opt_threshold',0.5)):>8.3f}"
                  f"  {str(row['selected_input_vars'])}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config: TrainConfig) -> Dict[str, Any]:
    set_seed(config.random_state)
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    data_dir = resolve_data_dir(config, script_dir)
    save_dir = resolve_save_dir(config, script_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device = configure_torch(config); config = sanitize_runtime_config(config, device)
    print(f"Data : {data_dir}  |  Save : {save_dir}  |  Device : {device}")
    if device.type == "cuda":
        print(f"GPU  : {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")

    runs = load_runs_from_npz_dir(data_dir, config=config, min_run_length=config.min_run_length)
    print_dataset_summary(runs)

    runs_by_vehicle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs: runs_by_vehicle[str(run["vehicle_type"])].append(run)

    total_trainings = estimate_total_trainings(runs_by_vehicle, config)
    print(f"Estimated model trainings: {total_trainings}")
    tracker = ProgressTracker(total_trainings, heartbeat_seconds=config.heartbeat_seconds)

    overall_rows: List[Dict[str, Any]] = []; all_results: Dict[str, Any] = {}
    for vehicle_type in sorted(runs_by_vehicle.keys()):
        vruns = runs_by_vehicle[vehicle_type]
        if len({int(r["label"]) for r in vruns}) < 2: print(f"Skipping {vehicle_type}: one class"); continue
        if len(get_unique_pilot_ids(vruns)) < 3: print(f"Skipping {vehicle_type}: <3 pilots"); continue
        result = run_vehicle_experiment(vruns, vehicle_type, config, device, tracker=tracker)
        all_results[vehicle_type] = result
        overall_rows.append(save_vehicle_results(result, save_dir))
        print_summary(result)

    if overall_rows:
        write_csv(save_dir / "overall_results.csv", overall_rows)
        _save_plot_overall(overall_rows, save_dir / "overall_accuracy.png")
        print("\n" + "=" * 72 + "\nFINAL SUMMARY")
        for row in overall_rows:
            print(f"  {row['vehicle_type']}: accuracy={row['run_level_accuracy']:.3f}  "
                  f"threshold={row['mean_threshold']:.3f}")
        print("=" * 72)
    return all_results


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    config = TrainConfig(
        data_dir=r"C:\\Users\\bramb\\Downloads\\AI_project_simulator\\A02-TAS-Repository\\data\\python_data",
        save_dir=r"C:\\Users\\bramb\\Downloads\\AI_project_simulator\\A02-TAS-Repository\\results_lstm_v5",

        window_sizes=(128, 256, 512, 1024),
        stride_fraction=0.5,

        input_combinations=(
            ("e", "u"),
            ("e", "u", "de", "du"),
            ("e", "u", "de", "du", "d2e", "d2u"),
            ("e", "u", "e_vlo", "u_vlo", "e_lo", "u_lo"),
            ("e", "u", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
            ("e", "u", "eu", "u_de", "e_du"),
            ("e", "u", "eu", "u_de", "e_du", "de", "du"),
            ("e", "u", "de", "du", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
            ("e", "u", "eu", "u_de", "e_du", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
            (
                "e", "u", "eu", "u_de", "e_du", "de", "du",
                "e_vlo", "u_vlo", "e_lo", "u_lo",
                "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr",
                "e_logstd", "u_logstd",
            ),
        ),

        batch_size=256 if use_cuda else 64,
        eval_batch_size=2048 if use_cuda else 256,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        grad_clip_norm=1.0,

        # FIX 2: capped weights — prevent over-correction
        class_weight_min=0.5,
        class_weight_max=2.0,

        # FIX 1: 30% of training windows used for per-fold threshold calibration
        threshold_cal_fraction=0.30,

        learning_rate=5e-4,
        weight_decay=1e-4,
        max_epochs=40,
        patience=8,
        val_check_interval=2,

        stage1_enabled=True,
        stage1_epochs=8,
        stage1_patience=3,
        stage1_top_k=6,
        global_tune_once_per_vehicle=True,

        random_state=42,
        min_run_length=256,
        default_fs=100.0,

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
