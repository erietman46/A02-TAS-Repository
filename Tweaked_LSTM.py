"""
lstm_simple.py
==============
A clean, readable BiLSTM for motion-sickness detection from simulator signals.

Structure
---------
1.  Config            — all hyperparameters in one place
2.  Data loading      — read .npz files, infer labels from filenames
3.  Feature engineering — derivatives and bandpass channels
4.  Windowing         — slice each run into fixed-length windows
5.  Dataset           — PyTorch Dataset wrapper
6.  Model             — plain BiLSTM classifier
7.  Training          — one-epoch loop with class-balanced loss + progress bar
8.  Evaluation        — accuracy, confusion matrix, per-class bias
9.  Cross-validation  — leave-one-pilot-out outer loop
10. Main              — runs all input combinations and prints final tables
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # --- paths ---
    data_dir: str = "data/python_data"
    save_dir: str = "results_lstm_simple"

    # --- windowing ---
    # 5-second windows at 100 Hz = 500 samples, 75 % overlap → stride 125.
    # The reference group found this outperforms shorter windows.
    window_size: int = 500   # 5 s × 100 Hz
    stride: int      = 125   # 75 % overlap
    fs: float        = 100.0

    # --- input combinations to benchmark ---
    # Tested in order; results are compared in the final summary table.
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
        ("e", "u", "eu", "u_de", "e_du", "de", "du",
         "e_vlo", "u_vlo", "e_lo", "u_lo",
         "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr",
         "e_logstd", "u_logstd"),
    )

    # --- model ---
    hidden_size: int  = 64
    num_layers: int   = 2
    dropout: float    = 0.3

    # --- training ---
    # Grid-search from the reference group found lr=0.01 and wd=1e-9 optimal.
    learning_rate: float = 1e-2
    weight_decay: float  = 1e-9
    max_epochs: int      = 50
    patience: int        = 8
    batch_size: int      = 128

    # --- misc ---
    seed: int   = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    file_pattern: str = (
        r"^ae2224I_measurement_data_subj(?P<subj>[1-6])_(?P<cond>C[1-6])\.npz$"
    )
    condition_label: Dict[str, int] = {
        "C1": 0, "C2": 0, "C3": 0, "C4": 1, "C5": 1, "C6": 1,
    }
    condition_vehicle: Dict[str, str] = {
        "C1": "P", "C2": "V", "C3": "A",
        "C4": "P", "C5": "V", "C6": "A",
    }


cfg = Config()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(data_dir: str, cfg: Config) -> List[dict]:
    """Scan data_dir for .npz files and return a list of run dicts."""
    pattern = re.compile(cfg.file_pattern, re.IGNORECASE)
    runs: List[dict] = []

    for npz_path in sorted(Path(data_dir).glob("*.npz")):
        m = pattern.match(npz_path.name)
        if m is None:
            continue

        condition = m.group("cond").upper()
        subject   = f"subj{m.group('subj')}"
        label     = cfg.condition_label[condition]
        vehicle   = cfg.condition_vehicle[condition]

        with np.load(npz_path, allow_pickle=True) as npz:
            e_list = _split_signal(npz["e"])
            u_list = _split_signal(npz["u"])

        if len(e_list) != len(u_list):
            raise ValueError(
                f"{npz_path.name}: e has {len(e_list)} runs but u has {len(u_list)}"
            )

        for rep, (e_r, u_r) in enumerate(zip(e_list, u_list), 1):
            n = min(len(e_r), len(u_r))
            if n < cfg.window_size:
                continue  # too short to form even one window
            runs.append(_make_run(e_r[:n], u_r[:n], label, vehicle, subject, condition, rep))

    if not runs:
        raise FileNotFoundError(f"No matching .npz files found in {data_dir}")
    return runs


def _split_signal(raw) -> List[np.ndarray]:
    """
    Convert a raw npz value into a list of 1-D float32 arrays, one per run.

    Handles the common MATLAB / npz export layouts
    -----------------------------------------------
    Object array (MATLAB cell)     each element is one run
    1-D array                      single run
    2-D tall  (samples > runs)     columns are runs  e.g. shape (8192, 25)
    2-D wide  (runs > samples)     rows are runs     e.g. shape (25, 8192)
    """
    arr = np.asarray(raw)

    # Object array — each flat element is one run (MATLAB cell export)
    if arr.dtype == object:
        return [np.asarray(arr.flat[i], dtype=np.float32).reshape(-1)
                for i in range(arr.size)]

    arr = arr.squeeze()

    if arr.ndim == 0:
        return [arr.reshape(1).astype(np.float32)]

    if arr.ndim == 1:
        return [arr.astype(np.float32)]

    # 2-D: the longer axis is samples, the shorter axis is runs.
    # Typical MATLAB layout exports as (N_samples, N_runs) — a tall matrix.
    if arr.shape[0] >= arr.shape[1]:
        # tall matrix → columns are individual runs
        return [arr[:, i].astype(np.float32) for i in range(arr.shape[1])]
    else:
        # wide matrix → rows are individual runs
        return [arr[i, :].astype(np.float32) for i in range(arr.shape[0])]


def _make_run(e, u, label, vehicle, subject, condition, rep) -> dict:
    return {
        "e": e, "u": u,
        "label": label, "vehicle": vehicle,
        "pilot": subject, "condition": condition,
        "run_id": f"{subject}_{condition}_rep{rep}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(run: dict, cfg: Config) -> dict:
    """
    Compute all derived channels and store them on a copy of the run dict.

    Channel       Description
    ───────────── ──────────────────────────────────────────────────────
    e, u          raw originals — mean-centred ONLY, NOT z-scored.
                  Amplitude carries information (paper: "normalisation
                  was not necessary").
    de, du        first derivatives (z-scored)
    d2e, d2u      second derivatives (z-scored)
    eu            normalised cross product  e×u
    u_de          cross term  u × de  (z-scored)
    e_du          cross term  e × du  (z-scored)
    e_vlo/u_vlo   0.05–0.5 Hz bandpass (z-scored)
    e_lo/u_lo     0.5–3.0 Hz bandpass  (z-scored)
    e_vlo_pwr     instantaneous power of e_vlo (z-scored)
    u_vlo_pwr     instantaneous power of u_vlo (z-scored)
    e_lo_pwr      instantaneous power of e_lo  (z-scored)
    u_lo_pwr      instantaneous power of u_lo  (z-scored)
    e_logstd      run-level log(1 + std(e)), broadcast to full length
    u_logstd      run-level log(1 + std(u)), broadcast to full length
    """
    run = dict(run)
    raw_e = np.asarray(run["e"], np.float32)
    raw_u = np.asarray(run["u"], np.float32)
    n     = len(raw_e)

    # Mean-centre only — preserve amplitude scale
    e = (raw_e - raw_e.mean()).astype(np.float32)
    u = (raw_u - raw_u.mean()).astype(np.float32)
    run["e"] = e
    run["u"] = u

    # Derivatives
    dt   = 1.0 / cfg.fs
    de   = np.gradient(e, dt).astype(np.float32)
    du   = np.gradient(u, dt).astype(np.float32)
    run["de"]  = _zscore(de)
    run["du"]  = _zscore(du)
    run["d2e"] = _zscore(np.gradient(de, dt).astype(np.float32))
    run["d2u"] = _zscore(np.gradient(du, dt).astype(np.float32))

    # Cross-signal features
    std_e = float(np.std(e)) + 1e-8
    std_u = float(np.std(u)) + 1e-8
    run["eu"]   = (e * u / (std_e * std_u)).astype(np.float32)
    run["u_de"] = _zscore((u * de).astype(np.float32))
    run["e_du"] = _zscore((e * du).astype(np.float32))

    # Bandpass channels
    e_vlo = _bandpass(raw_e, cfg.fs, 0.05, 0.5)
    u_vlo = _bandpass(raw_u, cfg.fs, 0.05, 0.5)
    e_lo  = _bandpass(raw_e, cfg.fs, 0.5,  3.0)
    u_lo  = _bandpass(raw_u, cfg.fs, 0.5,  3.0)
    run["e_vlo"] = _zscore(e_vlo)
    run["u_vlo"] = _zscore(u_vlo)
    run["e_lo"]  = _zscore(e_lo)
    run["u_lo"]  = _zscore(u_lo)

    # Instantaneous spectral power
    run["e_vlo_pwr"] = _zscore((e_vlo ** 2).astype(np.float32))
    run["u_vlo_pwr"] = _zscore((u_vlo ** 2).astype(np.float32))
    run["e_lo_pwr"]  = _zscore((e_lo  ** 2).astype(np.float32))
    run["u_lo_pwr"]  = _zscore((u_lo  ** 2).astype(np.float32))

    # Run-level log-std (scalar amplitude marker, broadcast to full length)
    run["e_logstd"] = np.full(n, float(np.log1p(np.std(raw_e))), np.float32)
    run["u_logstd"] = np.full(n, float(np.log1p(np.std(raw_u))), np.float32)

    return run


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-8 else 1.0)


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    if not HAS_SCIPY:
        return np.zeros_like(x)
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return np.zeros_like(x)
    return filtfilt(b, a, x).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WINDOWING
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(runs: List[dict], cfg: Config) -> List[dict]:
    """Slice every run into overlapping fixed-length windows (index-only, lazy)."""
    windows: List[dict] = []
    for run in runs:
        n = len(run["e"])
        if n < cfg.window_size:
            continue
        for start in range(0, n - cfg.window_size + 1, cfg.stride):
            windows.append({
                "run_id":  run["run_id"],
                "pilot":   run["pilot"],
                "vehicle": run["vehicle"],
                "label":   run["label"],
                "start":   start,
                "end":     start + cfg.window_size,
            })
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    x shape : (window_size, n_features)
    y       : 0 = no motion,  1 = motion
    """
    def __init__(self, windows, run_lookup, input_vars):
        self.windows    = windows
        self.run_lookup = run_lookup
        self.input_vars = input_vars

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w   = self.windows[idx]
        run = self.run_lookup[w["run_id"]]
        s, e = w["start"], w["end"]
        x = np.stack([run[v][s:e] for v in self.input_vars], axis=-1).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(w["label"], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL — Plain BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Input  : (batch, seq_len, n_features)
    Output : (batch, 2)   raw logits for [no-motion, motion]

    BiLSTM -> last hidden state -> LayerNorm -> Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = True,
        )
        h2 = hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(h2),
            nn.Linear(h2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (batch, seq, 2*hidden)
        return self.head(out[:, -1, :]) # use the final time step


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING  (with progress bar)
# ─────────────────────────────────────────────────────────────────────────────

class ProgressBar:
    """
    Single-line updating ASCII progress bar. No extra dependencies.

    Usage
    -----
        bar = ProgressBar(total=30, prefix="Training")
        for epoch in range(30):
            ...
            bar.update(epoch + 1, suffix="val=0.82")
        bar.close()
    """
    WIDTH = 28

    def __init__(self, total: int, prefix: str = ""):
        self.total      = max(1, total)
        self.prefix     = prefix
        self._start     = time.time()
        self._last_draw = 0.0
        self.update(0)

    def update(self, n: int, suffix: str = "") -> None:
        now = time.time()
        # Redraw at most 10 times/second to avoid slowing fast loops
        if n < self.total and now - self._last_draw < 0.1:
            return
        self._last_draw = now

        frac    = min(n / self.total, 1.0)
        filled  = int(self.WIDTH * frac)
        bar     = "█" * filled + "░" * (self.WIDTH - filled)
        elapsed = now - self._start
        eta     = (elapsed / frac - elapsed) if frac > 1e-6 else 0.0

        line = (
            f"\r  {self.prefix}  [{bar}] {frac*100:5.1f}%"
            f"  {n}/{self.total}"
            f"  {_fmt_time(elapsed)} elapsed"
            f"  ETA {_fmt_time(eta)}"
            + (f"  {suffix}" if suffix else "")
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def close(self, msg: str = "") -> None:
        elapsed = time.time() - self._start
        bar     = "█" * self.WIDTH
        sys.stdout.write(
            f"\r  {self.prefix}  [{bar}] 100.0%"
            f"  {self.total}/{self.total}"
            f"  {_fmt_time(elapsed)} elapsed"
            + (f"  {msg}" if msg else "")
            + "\n"
        )
        sys.stdout.flush()


def _fmt_time(s: float) -> str:
    if not (0 <= s < 36000):
        return "--:--"
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def _class_weights(windows: List[dict], device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights:  weight_c = n_total / (2 * n_c)
    Balances the loss so that motion/no-motion imbalance does not bias training.
    """
    counts = Counter(w["label"] for w in windows)
    n      = len(windows)
    return torch.tensor(
        [n / (2 * max(counts[0], 1)),
         n / (2 * max(counts[1], 1))],
        dtype=torch.float32, device=device,
    )


def _make_loader(windows, run_lookup, input_vars, batch_size, shuffle) -> DataLoader:
    return DataLoader(
        WindowDataset(windows, run_lookup, input_vars),
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
    )


def train_model(
    train_windows: List[dict],
    val_windows:   List[dict],
    run_lookup:    Dict[str, dict],
    input_vars:    Tuple[str, ...],
    cfg:           Config,
    fold_label:    str = "",
) -> nn.Module:
    """
    Train a BiLSTM with early stopping.
    Shows a progress bar that updates each epoch.
    Returns the best checkpoint (highest validation accuracy).
    """
    device = torch.device(cfg.device)

    model = BiLSTMClassifier(
        input_size  = len(input_vars),
        hidden_size = cfg.hidden_size,
        num_layers  = cfg.num_layers,
        dropout     = cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_windows, device))
    # Adam with lr=0.01 and very small weight-decay — optimal per reference group grid search.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=1e-5
    )

    train_loader = _make_loader(train_windows, run_lookup, input_vars, cfg.batch_size, True)
    val_loader   = _make_loader(val_windows,   run_lookup, input_vars, cfg.batch_size, False)

    best_val_acc  = -1.0
    best_state    = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience

    bar = ProgressBar(total=cfg.max_epochs, prefix=fold_label)

    for epoch in range(1, cfg.max_epochs + 1):

        # ── train one epoch ───────────────────────────────────────────────
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────
        val_acc = _evaluate(model, val_loader, device)
        bar.update(epoch, suffix=f"val={val_acc:.3f}  best={best_val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                bar.close(msg=f"early stop  best={best_val_acc:.3f}")
                model.load_state_dict(best_state)
                return model

    bar.close(msg=f"done  best={best_val_acc:.3f}")
    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Fraction of correctly classified windows (used during training)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        preds    = model(x.to(device)).argmax(dim=1).cpu()
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total if total else 0.0


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred) arrays for the whole loader."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, per-class rates, bias and confusion matrix."""
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    neg, pos = tn + fp, fn + tp

    spec = tn / neg if neg > 0 else 0.0  # no-motion accuracy
    sens = tp / pos if pos > 0 else 0.0  # motion accuracy
    fpr  = fp / neg if neg > 0 else 0.0
    fnr  = fn / pos if pos > 0 else 0.0
    bias = fpr - fnr

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "spec": float(spec), "sens": float(sens),
        "fpr":  float(fpr),  "fnr":  float(fnr),
        "bias": float(bias),
        "bias_direction": (
            "motion bias"    if bias >  0.05 else
            "no-motion bias" if bias < -0.05 else
            "balanced"
        ),
        "cm": cm,
    }


def print_vehicle_report(vehicle: str, metrics: dict, input_vars: Tuple[str, ...]) -> None:
    """Box-formatted accuracy + confusion matrix + bias matrix for one vehicle/combo."""
    cm  = metrics["cm"]
    W   = 56

    def row(left, right=""):
        inner = f"  {left:<28}  {right}"
        return "║" + inner + " " * max(0, W - len(inner)) + "║"

    print()
    print("╔" + "═" * W + "╗")
    print("║" + f"  Vehicle {vehicle}   Features: {' + '.join(input_vars)}"[:W].ljust(W) + "║")
    print("╠" + "═" * W + "╣")

    print(row("Overall accuracy", f"{metrics['accuracy']:>8.1%}"))
    print("╠" + "═" * W + "╣")

    # Confusion matrix
    print("║" + "  Confusion matrix".ljust(W) + "║")
    print("║" + f"  {'':>22}  {'Pred: no-motion':>14}  {'Pred: motion':>11}  "[:W].ljust(W) + "║")
    print("║" + f"  {'Actual: no-motion':>22}  {cm[0,0]:>14}  {cm[0,1]:>11}  "[:W].ljust(W) + "║")
    print("║" + f"  {'Actual: motion':>22}  {cm[1,0]:>14}  {cm[1,1]:>11}  "[:W].ljust(W) + "║")
    print("╠" + "═" * W + "╣")

    # Bias matrix
    print("║" + "  Bias matrix".ljust(W) + "║")
    hdr = f"  {'Class':<18}  {'Accuracy':>9}  {'Error rate':>10}  {'n':>5}"
    print("║" + hdr[:W].ljust(W) + "║")
    r0 = f"  {'No-motion':<18}  {metrics['spec']:>9.3f}  {metrics['fpr']:>10.3f}  {cm[0,0]+cm[0,1]:>5}"
    r1 = f"  {'Motion':<18}  {metrics['sens']:>9.3f}  {metrics['fnr']:>10.3f}  {cm[1,0]+cm[1,1]:>5}"
    print("║" + r0[:W].ljust(W) + "║")
    print("║" + r1[:W].ljust(W) + "║")
    print("╠" + "═" * W + "╣")

    bias_line = f"  Bias (FPR - FNR): {metrics['bias']:+.3f}  ->  {metrics['bias_direction']}"
    print("║" + bias_line[:W].ljust(W) + "║")
    print("╚" + "═" * W + "╝")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CROSS-VALIDATION — leave-one-pilot-out
# ─────────────────────────────────────────────────────────────────────────────

def run_vehicle(
    runs:       List[dict],
    vehicle:    str,
    input_vars: Tuple[str, ...],
    cfg:        Config,
) -> dict:
    """
    Leave-one-pilot-out CV for a single vehicle + input-combination.
    Predictions from all held-out pilots are pooled before computing metrics.
    """
    pilots = sorted({r["pilot"] for r in runs})
    if len(pilots) < 2:
        raise ValueError(f"{vehicle}: need >=2 pilots, got {len(pilots)}")
    if len({r["label"] for r in runs}) < 2:
        raise ValueError(f"{vehicle}: only one class present")

    run_lookup = {r["run_id"]: engineer_features(r, cfg) for r in runs}
    device     = torch.device(cfg.device)

    all_y_true, all_y_pred = [], []

    for i, test_pilot in enumerate(pilots, 1):
        fold_label = f"[{vehicle}] fold {i}/{len(pilots)} test={test_pilot}"
        print(f"\n  {fold_label}")

        train_runs = [r for r in runs if r["pilot"] != test_pilot]
        test_runs  = [r for r in runs if r["pilot"] == test_pilot]

        train_windows = build_windows(train_runs, cfg)
        test_windows  = build_windows(test_runs,  cfg)

        if not train_windows or not test_windows:
            print("  Skipping — insufficient windows")
            continue

        model = train_model(
            train_windows, test_windows, run_lookup, input_vars, cfg,
            fold_label=fold_label,
        )

        loader = _make_loader(test_windows, run_lookup, input_vars, cfg.batch_size, False)
        y_true, y_pred = predict(model, loader, device)
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    return compute_metrics(
        np.array(all_y_true, dtype=int),
        np.array(all_y_pred, dtype=int),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _print_combo_table(combo_results: List[dict]) -> None:
    """Ranked table of accuracy and bias for every combination x vehicle."""
    vehicles = sorted({r["vehicle"] for r in combo_results})
    combos   = list(dict.fromkeys(r["combo_label"] for r in combo_results))
    idx      = {(r["combo_label"], r["vehicle"]): r for r in combo_results}

    col = 10
    lw  = max(len(c) for c in combos) + 2
    sep = "─" * (lw + 4 + (col * 2 + 5) * len(vehicles))

    print()
    print("  ACCURACY & BIAS PER INPUT COMBINATION")
    print(f"  {sep}")
    hdr = f"  {'Input combination':<{lw}}"
    for v in vehicles:
        hdr += f"  {'Acc-' + v:>{col}}  {'Bias-' + v:>{col}}"
    print(hdr)
    print(f"  {sep}")

    def mean_acc(combo):
        vals = [idx[(combo, v)]["accuracy"] for v in vehicles if (combo, v) in idx]
        return sum(vals) / len(vals) if vals else 0.0

    for combo in sorted(combos, key=mean_acc, reverse=True):
        row = f"  {combo:<{lw}}"
        for v in vehicles:
            if (combo, v) in idx:
                r = idx[(combo, v)]
                row += f"  {r['accuracy']:>{col}.3f}  {r['bias']:>+{col}.3f}"
            else:
                row += f"  {'N/A':>{col}}  {'':>{col}}"
        print(row)

    print(f"  {sep}")


def main():
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  BiLSTM Motion Detection")
    print(f"  Device  : {cfg.device}")
    print(f"  Window  : {cfg.window_size} samples   stride {cfg.stride}")
    print(f"  Combos  : {len(cfg.input_combinations)} input combination(s) to test")
    print("=" * 62)

    runs = load_runs(cfg.data_dir, cfg)
    by_vehicle: Dict[str, List[dict]] = defaultdict(list)
    for r in runs:
        by_vehicle[r["vehicle"]].append(r)

    pilots = sorted({r["pilot"] for r in runs})
    print(f"\n  Loaded {len(runs)} runs  |  pilots: {', '.join(pilots)}")
    print(f"  Vehicles: {sorted(by_vehicle)}")
    if not HAS_SCIPY:
        print("  WARNING: scipy not found — e_bp/u_bp will be zeros (pip install scipy)")

    # Count total folds for overall-progress display
    total_folds = (
        sum(len({r["pilot"] for r in v}) for v in by_vehicle.values())
        * len(cfg.input_combinations)
    )
    folds_done = 0

    combo_results: List[dict] = []
    json_output:   dict       = {}

    for ci, input_vars in enumerate(cfg.input_combinations, 1):
        combo_label = " + ".join(input_vars)
        print()
        print(f"╔{'═'*60}╗")
        print(f"║  Combination {ci}/{len(cfg.input_combinations)}: {combo_label:<{46}}║")
        print(f"╚{'═'*60}╝")

        combo_json: dict = {}
        for vehicle in sorted(by_vehicle):
            vruns    = by_vehicle[vehicle]
            n_pilots = len({r["pilot"] for r in vruns})
            folds_done += n_pilots

            pct = folds_done / max(total_folds, 1) * 100
            print(f"\n  Vehicle {vehicle}  "
                  f"({len(vruns)} runs, {n_pilots} pilots)  "
                  f"[overall: {pct:.0f}% of all folds]")

            metrics = run_vehicle(vruns, vehicle, input_vars, cfg)
            print_vehicle_report(vehicle, metrics, input_vars)

            combo_results.append({
                "combo_label": combo_label,
                "vehicle":     vehicle,
                **{k: v for k, v in metrics.items() if k != "cm"},
            })
            combo_json[vehicle] = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in metrics.items()
            }

        json_output[combo_label] = combo_json

    # ── final summary tables ──────────────────────────────────────────────────
    _print_combo_table(combo_results)

    vehicles = sorted(by_vehicle)
    print()
    print("  BEST INPUT COMBINATION PER VEHICLE")
    print(f"  {'─'*62}")
    print(f"  {'Vehicle':<10}  {'Best combination':<30}  {'Accuracy':>9}  {'Bias':>8}")
    print(f"  {'─'*62}")
    for v in vehicles:
        v_rows = [r for r in combo_results if r["vehicle"] == v]
        if not v_rows:
            continue
        best = max(v_rows, key=lambda r: r["accuracy"])
        print(f"  {v:<10}  {best['combo_label']:<30}  {best['accuracy']:>9.3f}  {best['bias']:>+8.3f}")
    print(f"  {'─'*62}")

    # Save JSON
    out_path = save_path / "results.json"
    out_path.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\n  Results saved -> {out_path}")

    # Optional grouped bar chart
    if HAS_MPL and combo_results:
        _plot_results(combo_results, vehicles, save_path)

    return json_output


def _plot_results(combo_results, vehicles, save_path) -> None:
    combos  = list(dict.fromkeys(r["combo_label"] for r in combo_results))
    n_v     = len(vehicles)
    x       = np.arange(len(combos))
    width   = 0.8 / n_v
    colors  = ["#3a86ff", "#ff006e", "#8338ec", "#06d6a0"]
    idx     = {(r["combo_label"], r["vehicle"]): r for r in combo_results}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, v in enumerate(vehicles):
        accs   = [idx.get((c, v), {}).get("accuracy", 0.0) for c in combos]
        biases = [idx.get((c, v), {}).get("bias",     0.0) for c in combos]
        kw     = dict(width=width, color=colors[i % len(colors)], label=f"Vehicle {v}")
        ax1.bar(x + i * width, accs,   **kw)
        ax2.bar(x + i * width, biases, **kw)

    for ax in (ax1, ax2):
        ax.set_xticks(x + width * (n_v - 1) / 2)
        ax.set_xticklabels(combos, rotation=18, ha="right", fontsize=9)
        ax.legend(fontsize=9)

    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by input combination & vehicle")
    ax1.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, label="80% target")

    ax2.axhline(0,     color="black", linewidth=0.8)
    ax2.axhline(0.05,  color="red",   linestyle="--", linewidth=0.8, label="bias threshold")
    ax2.axhline(-0.05, color="red",   linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Bias  (FPR - FNR)")
    ax2.set_title("Bias by input combination & vehicle")

    fig.tight_layout()
    out = save_path / "combo_accuracy.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  Plot saved  -> {out}")


if __name__ == "__main__":
    # ── config overrides ──────────────────────────────────────────────────────
    cfg.data_dir = r"C:\Users\bramb\Downloads\AI_project_simulator\A02-TAS-Repository\data\python_data"
    cfg.save_dir = r"C:\Users\bramb\Downloads\AI_project_simulator\A02-TAS-Repository\results_lstm_simple"

    cfg.input_combinations = (
        ("e", "u"),
        ("e", "u", "de", "du"),
        ("e", "u", "de", "du", "d2e", "d2u"),
        ("e", "u", "e_vlo", "u_vlo", "e_lo", "u_lo"),
        ("e", "u", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
        ("e", "u", "eu", "u_de", "e_du"),
        ("e", "u", "eu", "u_de", "e_du", "de", "du"),
        ("e", "u", "de", "du", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
        ("e", "u", "eu", "u_de", "e_du", "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr"),
        ("e", "u", "eu", "u_de", "e_du", "de", "du",
         "e_vlo", "u_vlo", "e_lo", "u_lo",
         "e_vlo_pwr", "u_vlo_pwr", "e_lo_pwr", "u_lo_pwr",
         "e_logstd", "u_logstd"),
    )

    # 5-second windows, 75 % overlap — matches the reference group's setup
    cfg.window_size   = 500
    cfg.stride        = 125

    # Hyperparameters from reference group grid search
    cfg.learning_rate = 1e-2
    cfg.weight_decay  = 1e-9
    cfg.max_epochs    = 50
    cfg.patience      = 8
    cfg.batch_size    = 128

    cfg.hidden_size   = 64
    cfg.num_layers    = 2
    cfg.dropout       = 0.3
    # ─────────────────────────────────────────────────────────────────────────
    main()