from __future__ import annotations

"""
LSTM classifier for motion-on vs motion-off behavior in the AE2224-I dataset.

Dataset assumptions
-------------------
Folder layout expected by default:
    A02-TAS-Repository/
      LSTM_model.py
      data/
        python_data/
          ae2224I_measurement_data_subj1_C1.npz
          ...
          ae2224I_measurement_data_subj6_C6.npz

File naming convention:
    ae2224I_measurement_data_subj<1-6>_C<1-6>.npz

Condition mapping:
    C1 = Gain (P), no motion
    C2 = Single integrator (V), no motion
    C3 = Double integrator (A), no motion
    C4 = Gain (P), motion
    C5 = Single integrator (V), motion
    C6 = Double integrator (A), motion

Expected signal keys inside each .npz:
    e, u, and optionally t

Each .npz may contain multiple runs, for example MATLAB-style object arrays of shape (20, 1).
Each contained run becomes one repetition.
"""

import copy
import csv
import json
import math
import time
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None


@dataclass
class TrainConfig:
    data_dir: Optional[str | Path] = None
    save_dir: Optional[str | Path] = None
    window_sizes: Tuple[int, ...] = (64, 96)
    stride_fraction: float = 1.0
    input_combinations: Tuple[Tuple[str, ...], ...] = (
        ("e", "u"),
        ("e", "u", "de", "du"),
    )
    batch_size: int = 256
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 4
    screen_max_epochs: int = 6
    screen_patience: int = 2
    screen_num_folds: int = 2
    top_k_candidates: int = 1
    max_windows_per_run: Optional[int] = 24
    precompute_windows: bool = True
    random_state: int = 42
    min_run_length: int = 128
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    device: str = "auto"
    filename_pattern: str = r"^ae2224I_measurement_data_subj(?P<subject>[1-6])_(?P<condition>C[1-6])\.npz$"
    signal_keys: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "e": ("e",),
        "u": ("u",),
        "time": ("t", "time"),
    })
    condition_to_label: Dict[str, int] = field(default_factory=lambda: {
        "C1": 0,
        "C2": 0,
        "C3": 0,
        "C4": 1,
        "C5": 1,
        "C6": 1,
    })
    condition_to_vehicle: Dict[str, str] = field(default_factory=lambda: {
        "C1": "P",
        "C2": "V",
        "C3": "A",
        "C4": "P",
        "C5": "V",
        "C6": "A",
    })


class ProgressTracker:
    def __init__(self, total_trainings: int):
        self.total_trainings = max(1, int(total_trainings))
        self.completed_trainings = 0
        self.start_time = time.time()
        self.current_label = ""
        self.current_planned_epochs = 1
        self.current_epoch = 0

    def start_training(self, label: str, planned_epochs: int) -> None:
        self.current_label = label
        self.current_planned_epochs = max(1, int(planned_epochs))
        self.current_epoch = 0
        self._print_status(prefix="START", newline=True)

    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))
        self._print_status(prefix="RUN", newline=False)

    def finish_training(self, actual_epochs: int) -> None:
        self.current_epoch = max(0, int(actual_epochs))
        self.completed_trainings += 1
        self._print_status(prefix="DONE", newline=True)

    def _fraction_complete(self) -> float:
        current_fraction = min(1.0, self.current_epoch / max(1, self.current_planned_epochs))
        return min(1.0, (self.completed_trainings + current_fraction) / self.total_trainings)

    def _format_seconds(self, seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--:--"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _status_message(self, prefix: str) -> str:
        elapsed = time.time() - self.start_time
        frac = self._fraction_complete()
        eta = (elapsed / frac - elapsed) if frac > 1e-9 else float('inf')
        return (
            f"[{prefix}] training {min(self.completed_trainings + 1, self.total_trainings)}/{self.total_trainings} | "
            f"overall {frac * 100:6.2f}% | epoch {self.current_epoch}/{self.current_planned_epochs} | "
            f"elapsed {self._format_seconds(elapsed)} | ETA {self._format_seconds(eta)} | "
            f"{self.current_label}"
        )

    def _print_status(self, prefix: str, newline: bool) -> None:
        message = self._status_message(prefix)
        if newline:
            print(message)
        else:
            print(message, end="\r", flush=True)




FILENAME_HELP = "ae2224I_measurement_data_subj<1-6>_C<1-6>.npz"


def resolve_device(requested_device: str) -> str:
    requested = str(requested_device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but this PyTorch installation cannot see a CUDA GPU. Falling back to CPU.")
        return "cpu"
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        print("MPS was requested but is not available. Falling back to CPU.")
        return "cpu"
    return requested


def configure_runtime(config: TrainConfig) -> None:
    config.device = resolve_device(config.device)
    if config.pin_memory is None:
        config.pin_memory = str(config.device).startswith("cuda")
    if str(config.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] = (0, 1)) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(labels)
    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            cm[idx[yt], idx[yp]] += 1
    return cm


def classification_report_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] = (0, 1)) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    report: Dict[str, Dict[str, float]] = {}
    supports: List[int] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        report[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    weights = np.asarray(supports, dtype=float)
    weight_sum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
    total_support = int(np.sum(supports))
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


def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-8:
        return x - mu
    return (x - mu) / sigma


def derivative(x: np.ndarray, time: Optional[np.ndarray] = None, fs: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if time is not None:
        time = np.asarray(time, dtype=np.float32).reshape(-1)
        if len(time) != len(x):
            raise ValueError("time must have same length as signal")
        return np.gradient(x, time).astype(np.float32)
    dt = 1.0 / fs if fs not in (None, 0) else 1.0
    return np.gradient(x, dt).astype(np.float32)


def normalize_run_signals(run: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(run)
    raw_e = np.asarray(out["e"], dtype=np.float32)
    raw_u = np.asarray(out["u"], dtype=np.float32)
    time = out.get("time")
    fs = out.get("fs")
    out["de"] = safe_zscore(derivative(raw_e, time=time, fs=fs))
    out["du"] = safe_zscore(derivative(raw_u, time=time, fs=fs))
    out["e"] = safe_zscore(raw_e)
    out["u"] = safe_zscore(raw_u)
    return out


def _first_available_key(npz_obj: Any, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for key in candidates:
        if key in npz_obj.files:
            return key
    if required:
        raise KeyError(f"None of the keys {tuple(candidates)} were found. Available keys: {npz_obj.files}")
    return None


def _to_1d_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.reshape(-1)[0])
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _split_npz_field_into_runs(value: Any) -> List[np.ndarray]:
    arr = np.asarray(value)

    if arr.dtype == object:
        return [_to_1d_float_array(item) for item in arr.reshape(-1)]

    if arr.ndim <= 1:
        return [_to_1d_float_array(arr)]

    if arr.ndim == 2 and 1 in arr.shape:
        flat = arr.reshape(-1)
        if flat.dtype == object:
            return [_to_1d_float_array(item) for item in flat]
        return [_to_1d_float_array(arr)]

    if arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]:
            return [_to_1d_float_array(arr[i, :]) for i in range(arr.shape[0])]
        return [_to_1d_float_array(arr[:, i]) for i in range(arr.shape[1])]

    return [_to_1d_float_array(arr)]


def _infer_fs_from_time(time_vector: Optional[np.ndarray]) -> Optional[float]:
    if time_vector is None or len(time_vector) < 2:
        return None
    dt = np.diff(np.asarray(time_vector, dtype=np.float32))
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return None
    median_dt = float(np.median(dt))
    if abs(median_dt) < 1e-12:
        return None
    return float(1.0 / median_dt)


def resolve_data_dir(config: TrainConfig, script_dir: Path) -> Path:
    candidates: List[Path] = []

    if config.data_dir not in (None, ""):
        p = Path(config.data_dir)
        candidates.append(p if p.is_absolute() else script_dir / p)
        candidates.append(p if p.is_absolute() else Path.cwd() / p)

    candidates.extend([
        script_dir / "data" / "python_data",
        script_dir / "python_data",
        Path.cwd() / "data" / "python_data",
        Path.cwd() / "python_data",
        script_dir.parent / "data" / "python_data",
        script_dir.parent / "python_data",
    ])

    seen: set[str] = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    checked = "\n".join(str(p) for p in unique_candidates)
    raise FileNotFoundError(
        "Could not find the data directory. Checked:\n"
        f"{checked}\n\n"
        "Expected to find your .npz files in a folder like: data/python_data"
    )


def resolve_save_dir(config: TrainConfig, script_dir: Path) -> Path:
    if config.save_dir in (None, ""):
        return script_dir / "results_lstm"
    p = Path(config.save_dir)
    return p if p.is_absolute() else script_dir / p


def load_runs_from_npz_dir(data_dir: str | Path, config: TrainConfig, min_run_length: int = 1) -> List[Dict[str, Any]]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    pattern = re.compile(config.filename_pattern, re.IGNORECASE)
    npz_files = [p for p in sorted(data_path.glob("*.npz")) if pattern.match(p.name)]
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files matched the expected pattern in {data_path}. Expected format: {FILENAME_HELP}"
        )

    expected = {f"ae2224I_measurement_data_subj{s}_C{c}.npz".lower() for s in range(1, 7) for c in range(1, 7)}
    found = {p.name.lower() for p in npz_files}
    missing = sorted(expected - found)
    if missing:
        print("Warning: missing expected files:")
        for name in missing:
            print(f"  - {name}")

    all_runs: List[Dict[str, Any]] = []
    for npz_file in npz_files:
        match = pattern.match(npz_file.name)
        if match is None:
            continue

        subject = f"subj{match.group('subject')}"
        condition = match.group('condition').upper()
        label = int(config.condition_to_label[condition])
        vehicle = str(config.condition_to_vehicle[condition])

        with np.load(npz_file, allow_pickle=True) as npz_obj:
            e_key = _first_available_key(npz_obj, config.signal_keys["e"], required=True)
            u_key = _first_available_key(npz_obj, config.signal_keys["u"], required=True)
            t_key = _first_available_key(npz_obj, config.signal_keys["time"], required=False)

            e_runs = _split_npz_field_into_runs(npz_obj[e_key])
            u_runs = _split_npz_field_into_runs(npz_obj[u_key])
            if len(e_runs) != len(u_runs):
                raise ValueError(f"{npz_file.name}: number of e runs ({len(e_runs)}) != number of u runs ({len(u_runs)})")

            t_runs: List[Optional[np.ndarray]]
            if t_key is None:
                t_runs = [None] * len(e_runs)
            else:
                raw_t_runs = _split_npz_field_into_runs(npz_obj[t_key])
                if len(raw_t_runs) == 1 and len(e_runs) > 1:
                    t_runs = [raw_t_runs[0] for _ in range(len(e_runs))]
                elif len(raw_t_runs) == len(e_runs):
                    t_runs = list(raw_t_runs)
                else:
                    raise ValueError(
                        f"{npz_file.name}: number of t runs ({len(raw_t_runs)}) does not match e runs ({len(e_runs)})"
                    )

            for rep_idx, (e_run, u_run, t_run) in enumerate(zip(e_runs, u_runs, t_runs), start=1):
                lengths = [len(e_run), len(u_run)]
                if t_run is not None:
                    lengths.append(len(t_run))
                n = min(lengths)
                if n < min_run_length:
                    continue

                run: Dict[str, Any] = {
                    "e": np.asarray(e_run[:n], dtype=np.float32),
                    "u": np.asarray(u_run[:n], dtype=np.float32),
                    "label": label,
                    "vehicle_type": vehicle,
                    "pilot_id": subject,
                    "condition_id": condition,
                    "repetition_id": f"rep{rep_idx}",
                    "source_file": npz_file.name,
                    "run_id": f"{npz_file.stem}_rep{rep_idx}",
                }
                if t_run is not None:
                    run["time"] = np.asarray(t_run[:n], dtype=np.float32)
                    fs = _infer_fs_from_time(run["time"])
                    if fs is not None:
                        run["fs"] = fs

                all_runs.append(normalize_run_signals(run))

    if not all_runs:
        raise ValueError("No valid runs were loaded. Check min_run_length and the contents of the .npz files.")
    return all_runs


class WindowDataset(Dataset):
    def __init__(self, windows: Sequence[Dict[str, Any]]):
        self.windows = list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        item = self.windows[idx]
        x = torch.tensor(item["x"], dtype=torch.float32)
        y = torch.tensor(item["y"], dtype=torch.long)
        meta = {
            "vehicle_type": item["vehicle_type"],
            "pilot_id": item["pilot_id"],
            "repetition_id": item["repetition_id"],
            "run_id": item["run_id"],
            "source_file": item["source_file"],
            "start_idx": item["start_idx"],
            "end_idx": item["end_idx"],
        }
        return x, y, meta


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int = 2):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)


def collate_fn(batch: Sequence[Any]):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def make_loader(windows: Sequence[Dict[str, Any]], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool = False) -> DataLoader:
    ds = WindowDataset(windows)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )


def evenly_subsample_windows(windows: List[Dict[str, Any]], max_windows: Optional[int]) -> List[Dict[str, Any]]:
    if max_windows is None or len(windows) <= max_windows:
        return windows
    idx = np.linspace(0, len(windows) - 1, int(max_windows), dtype=int)
    return [windows[int(i)] for i in idx]


def make_windows_for_run(run: Dict[str, Any], window_size: int, stride: int, input_vars: Sequence[str], max_windows_per_run: Optional[int] = None) -> List[Dict[str, Any]]:
    n = len(run["e"])
    out: List[Dict[str, Any]] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        x = np.stack([run[var][start:end] for var in input_vars], axis=-1)
        out.append({
            "x": x.astype(np.float32),
            "y": int(run["label"]),
            "vehicle_type": run["vehicle_type"],
            "pilot_id": run["pilot_id"],
            "repetition_id": run["repetition_id"],
            "run_id": run["run_id"],
            "source_file": run["source_file"],
            "start_idx": start,
            "end_idx": end,
        })
    return evenly_subsample_windows(out, max_windows_per_run)


def build_window_table(runs: Sequence[Dict[str, Any]], window_size: int, input_vars: Sequence[str], stride_fraction: float, max_windows_per_run: Optional[int] = None) -> List[Dict[str, Any]]:
    stride = max(1, int(round(window_size * stride_fraction)))
    all_windows: List[Dict[str, Any]] = []
    for run in runs:
        all_windows.extend(make_windows_for_run(run, window_size, stride, input_vars, max_windows_per_run=max_windows_per_run))
    if not all_windows:
        raise ValueError("No windows could be created. Check min_run_length and window_size.")
    return all_windows


def build_window_cache(runs: Sequence[Dict[str, Any]], candidate_specs: Sequence[Tuple[int, Tuple[str, ...]]], stride_fraction: float, max_windows_per_run: Optional[int]) -> Dict[Tuple[str, int, Tuple[str, ...]], List[Dict[str, Any]]]:
    cache: Dict[Tuple[str, int, Tuple[str, ...]], List[Dict[str, Any]]] = {}
    for window_size, input_vars in candidate_specs:
        stride = max(1, int(round(window_size * stride_fraction)))
        for run in runs:
            cache[(str(run["run_id"]), int(window_size), tuple(input_vars))] = make_windows_for_run(
                run,
                window_size,
                stride,
                input_vars,
                max_windows_per_run=max_windows_per_run,
            )
    return cache


def collect_windows(runs: Sequence[Dict[str, Any]], window_size: int, input_vars: Sequence[str], stride_fraction: float, max_windows_per_run: Optional[int], cache: Optional[Dict[Tuple[str, int, Tuple[str, ...]], List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    if cache is None:
        return build_window_table(runs, window_size, input_vars, stride_fraction, max_windows_per_run=max_windows_per_run)
    all_windows: List[Dict[str, Any]] = []
    key_vars = tuple(input_vars)
    for run in runs:
        all_windows.extend(cache.get((str(run["run_id"]), int(window_size), key_vars), []))
    if not all_windows:
        raise ValueError("No windows could be created. Check min_run_length and window_size.")
    return all_windows


def get_unique_pilot_ids(runs: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(r["pilot_id"]) for r in runs})


def split_runs_by_pilot(runs: Sequence[Dict[str, Any]], held_out_pilot: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_runs = [r for r in runs if str(r["pilot_id"]) != str(held_out_pilot)]
    test_runs = [r for r in runs if str(r["pilot_id"]) == str(held_out_pilot)]
    return train_runs, test_runs


def get_leave_one_pilot_out_folds(runs: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]]:
    return [(pilot_id, *split_runs_by_pilot(runs, pilot_id)) for pilot_id in get_unique_pilot_ids(runs)]


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y, _ in loader:
        non_blocking = str(device).startswith("cuda")
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    metas: List[Dict[str, Any]] = []
    for x, y, meta in loader:
        non_blocking = str(device).startswith("cuda")
        x = x.to(device, non_blocking=non_blocking)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs[:, 1].tolist())
        metas.extend(meta)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(y_prob, dtype=float), metas


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    y_true, y_pred, y_prob, metas = predict_loader(model, loader, device)
    return {
        "accuracy": accuracy_score_np(y_true, y_pred),
        "confusion_matrix": confusion_matrix_np(y_true, y_pred, labels=(0, 1)),
        "classification_report": classification_report_np(y_true, y_pred, labels=(0, 1)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metas": metas,
    }


def fit_lstm(train_windows: Sequence[Dict[str, Any]], val_windows: Sequence[Dict[str, Any]], config: TrainConfig, input_size: int, tracker: Optional[ProgressTracker] = None, training_label: str = ""):
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_windows, config.batch_size, True, config.num_workers, pin_memory=bool(config.pin_memory))
    val_loader = make_loader(val_windows, config.batch_size, False, config.num_workers, pin_memory=bool(config.pin_memory))

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -float("inf")
    best_epoch = 1
    history: List[Dict[str, Any]] = []
    wait = 0

    if tracker is not None:
        tracker.start_training(training_label or f"train_windows={len(train_windows)} val_windows={len(val_windows)}", config.max_epochs)

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_result = evaluate_model(model, val_loader, config.device)
        val_acc = float(val_result["accuracy"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})
        if tracker is not None:
            tracker.update_epoch(epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    model.load_state_dict(best_state)
    if tracker is not None:
        tracker.finish_training(actual_epochs=history[-1]["epoch"] if history else 0)
    return model, history, best_epoch, best_val_acc


def fit_lstm_fixed_epochs(train_windows: Sequence[Dict[str, Any]], config: TrainConfig, input_size: int, num_epochs: int, tracker: Optional[ProgressTracker] = None, training_label: str = ""):
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_windows, config.batch_size, True, config.num_workers, pin_memory=bool(config.pin_memory))
    history: List[Dict[str, Any]] = []

    planned_epochs = max(1, int(num_epochs))
    if tracker is not None:
        tracker.start_training(training_label or f"final_train_windows={len(train_windows)}", planned_epochs)

    for epoch in range(1, planned_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        history.append({"epoch": epoch, "train_loss": train_loss})
        if tracker is not None:
            tracker.update_epoch(epoch)

    if tracker is not None:
        tracker.finish_training(actual_epochs=planned_epochs)
    return model, history


def build_prediction_rows(eval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for yt, yp, prob, meta in zip(eval_result["y_true"], eval_result["y_pred"], eval_result["y_prob"], eval_result["metas"]):
        rows.append({
            "true_label": int(yt),
            "pred_label": int(yp),
            "motion_on_confidence": float(prob),
            "vehicle_type": meta["vehicle_type"],
            "pilot_id": meta["pilot_id"],
            "repetition_id": meta["repetition_id"],
            "run_id": meta["run_id"],
            "source_file": meta["source_file"],
            "start_idx": int(meta["start_idx"]),
            "end_idx": int(meta["end_idx"]),
        })
    return rows


def combine_prediction_rows(prediction_tables: Sequence[List[Dict[str, Any]]]) -> Dict[str, Any]:
    combined = [row for table in prediction_tables for row in table]
    if not combined:
        raise ValueError("No prediction rows to combine.")
    y_true = np.asarray([row["true_label"] for row in combined], dtype=int)
    y_pred = np.asarray([row["pred_label"] for row in combined], dtype=int)
    y_prob = np.asarray([row["motion_on_confidence"] for row in combined], dtype=float)
    return {
        "combined_prediction_rows": combined,
        "accuracy": accuracy_score_np(y_true, y_pred),
        "confusion_matrix": confusion_matrix_np(y_true, y_pred, labels=(0, 1)),
        "classification_report": classification_report_np(y_true, y_pred, labels=(0, 1)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def aggregate_run_confidence(pred_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, int], List[float]] = defaultdict(list)
    for row in pred_rows:
        key = (
            row["vehicle_type"],
            row["pilot_id"],
            row["repetition_id"],
            row["run_id"],
            int(row["true_label"]),
        )
        grouped[key].append(float(row["motion_on_confidence"]))

    out: List[Dict[str, Any]] = []
    for (vehicle_type, pilot_id, repetition_id, run_id, true_label), scores in grouped.items():
        out.append({
            "vehicle_type": vehicle_type,
            "pilot_id": pilot_id,
            "repetition_id": repetition_id,
            "run_id": run_id,
            "true_label": true_label,
            "mean_motion_on_confidence": float(np.mean(scores)),
            "num_windows": int(len(scores)),
        })
    return sorted(out, key=lambda r: (r["vehicle_type"], r["pilot_id"], r["repetition_id"], r["run_id"]))


def run_basic_statistics(pred_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    run_rows = aggregate_run_confidence(pred_rows)
    results: Dict[str, Any] = {"aggregated_run_predictions": run_rows}

    if stats is None:
        results["note"] = "scipy not installed; inferential statistics skipped"
        return results

    off_scores = np.asarray([r["mean_motion_on_confidence"] for r in run_rows if int(r["true_label"]) == 0], dtype=float)
    on_scores = np.asarray([r["mean_motion_on_confidence"] for r in run_rows if int(r["true_label"]) == 1], dtype=float)

    if len(off_scores) >= 2 and len(on_scores) >= 2:
        t_stat, p_val = stats.ttest_ind(off_scores, on_scores, equal_var=False)
        results["t_test_confidence_off_vs_on"] = {
            "n_off_runs": int(len(off_scores)),
            "n_on_runs": int(len(on_scores)),
            "mean_off": float(np.mean(off_scores)),
            "mean_on": float(np.mean(on_scores)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
        }

    by_pilot: Dict[str, List[float]] = defaultdict(list)
    for row in run_rows:
        by_pilot[str(row["pilot_id"])].append(float(row["mean_motion_on_confidence"]))
    pilot_groups = [np.asarray(vals, dtype=float) for vals in by_pilot.values() if len(vals) >= 2]
    if len(pilot_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*pilot_groups)
        results["anova_by_pilot"] = {
            "num_groups": int(len(pilot_groups)),
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
        }

    by_repetition: Dict[str, List[float]] = defaultdict(list)
    for row in run_rows:
        by_repetition[str(row["repetition_id"])].append(float(row["mean_motion_on_confidence"]))
    repetition_groups = [np.asarray(vals, dtype=float) for vals in by_repetition.values() if len(vals) >= 2]
    if len(repetition_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*repetition_groups)
        results["anova_by_repetition"] = {
            "num_groups": int(len(repetition_groups)),
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
        }

    return results


def estimate_total_trainings(runs_by_vehicle: Dict[str, List[Dict[str, Any]]], config: TrainConfig) -> int:
    total = 0
    num_candidates = len(config.window_sizes) * len(config.input_combinations)
    for vehicle_runs in runs_by_vehicle.values():
        labels = {int(r["label"]) for r in vehicle_runs}
        pilot_ids = get_unique_pilot_ids(vehicle_runs)
        if len(labels) < 2 or len(pilot_ids) < 3:
            continue
        n_pilots = len(pilot_ids)
        inner_folds = n_pilots - 1
        screen_folds = min(max(1, config.screen_num_folds), inner_folds)
        top_k = min(max(1, config.top_k_candidates), num_candidates)
        total += n_pilots * (num_candidates * screen_folds + top_k * inner_folds + 1)
    return total




def run_inner_cv_hyperparameter_search(
    train_runs: Sequence[Dict[str, Any]],
    vehicle_type: str,
    config: TrainConfig,
    tracker: Optional[ProgressTracker] = None,
    window_cache: Optional[Dict[Tuple[str, int, Tuple[str, ...]], List[Dict[str, Any]]]] = None,
):
    inner_folds = get_leave_one_pilot_out_folds(train_runs)
    if len(inner_folds) < 2:
        raise ValueError("Need at least two pilots in the outer-training split for inner CV.")

    all_candidates = [(tuple(input_vars), int(window_size)) for input_vars in config.input_combinations for window_size in config.window_sizes]
    screen_folds = inner_folds[: min(max(1, config.screen_num_folds), len(inner_folds))]

    screen_config = copy.copy(config)
    screen_config.max_epochs = min(config.max_epochs, max(1, config.screen_max_epochs))
    screen_config.patience = min(config.patience, max(1, config.screen_patience))

    screened_rows: List[Dict[str, Any]] = []
    viable_candidates: List[Tuple[Tuple[str, ...], int]] = []

    for input_vars, window_size in all_candidates:
        fold_accs: List[float] = []
        failed = False
        for val_pilot, inner_train_runs, val_runs in screen_folds:
            if not inner_train_runs or not val_runs or len({r["label"] for r in inner_train_runs}) < 2:
                failed = True
                break
            try:
                train_windows = collect_windows(inner_train_runs, window_size, input_vars, config.stride_fraction, config.max_windows_per_run, cache=window_cache)
                val_windows = collect_windows(val_runs, window_size, input_vars, config.stride_fraction, config.max_windows_per_run, cache=window_cache)
            except ValueError:
                failed = True
                break

            _, _, _, best_val_acc = fit_lstm(
                train_windows,
                val_windows,
                screen_config,
                input_size=len(input_vars),
                tracker=tracker,
                training_label=f"{vehicle_type} | screen | val={val_pilot} | vars={','.join(input_vars)} | w={window_size}",
            )
            fold_accs.append(float(best_val_acc))

        if failed or not fold_accs:
            continue

        viable_candidates.append((input_vars, window_size))
        screened_rows.append({
            "vehicle_type": vehicle_type,
            "input_vars": ",".join(input_vars),
            "window_size": int(window_size),
            "screen_mean_accuracy": float(np.mean(fold_accs)),
            "screen_std_accuracy": float(np.std(fold_accs, ddof=0)),
            "screen_num_folds": int(len(fold_accs)),
        })

    if not screened_rows:
        raise ValueError("Candidate screening failed for all hyperparameter settings.")

    screened_rows = sorted(screened_rows, key=lambda r: (-r["screen_mean_accuracy"], r["screen_std_accuracy"], r["window_size"]))
    top_k = min(max(1, config.top_k_candidates), len(screened_rows))
    selected_keys = {(row["input_vars"], int(row["window_size"])) for row in screened_rows[:top_k]}
    selected_candidates = [(vars_, w) for vars_, w in viable_candidates if (",".join(vars_), int(w)) in selected_keys]

    summary_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for input_vars, window_size in selected_candidates:
        candidate_fold_rows: List[Dict[str, Any]] = []
        failed = False

        for val_pilot, inner_train_runs, val_runs in inner_folds:
            if not inner_train_runs or not val_runs or len({r["label"] for r in inner_train_runs}) < 2:
                failed = True
                break
            try:
                train_windows = collect_windows(inner_train_runs, window_size, input_vars, config.stride_fraction, config.max_windows_per_run, cache=window_cache)
                val_windows = collect_windows(val_runs, window_size, input_vars, config.stride_fraction, config.max_windows_per_run, cache=window_cache)
            except ValueError:
                failed = True
                break

            _, _, best_epoch, best_val_acc = fit_lstm(
                train_windows,
                val_windows,
                config,
                input_size=len(input_vars),
                tracker=tracker,
                training_label=f"{vehicle_type} | inner | val={val_pilot} | vars={','.join(input_vars)} | w={window_size}",
            )
            candidate_fold_rows.append({
                "vehicle_type": vehicle_type,
                "validation_pilot": val_pilot,
                "input_vars": ",".join(input_vars),
                "window_size": int(window_size),
                "best_epoch": int(best_epoch),
                "best_val_accuracy": float(best_val_acc),
            })

        if failed or not candidate_fold_rows:
            continue

        fold_rows.extend(candidate_fold_rows)
        accs = [row["best_val_accuracy"] for row in candidate_fold_rows]
        epochs = [row["best_epoch"] for row in candidate_fold_rows]
        summary_rows.append({
            "vehicle_type": vehicle_type,
            "input_vars": ",".join(input_vars),
            "window_size": int(window_size),
            "mean_inner_val_accuracy": float(np.mean(accs)),
            "std_inner_val_accuracy": float(np.std(accs, ddof=0)),
            "mean_best_epoch": float(np.mean(epochs)),
            "num_inner_folds": int(len(candidate_fold_rows)),
            "screen_mean_accuracy": float(next(r["screen_mean_accuracy"] for r in screened_rows if r["input_vars"] == ",".join(input_vars) and int(r["window_size"]) == int(window_size))),
        })

    if not summary_rows:
        raise ValueError("Inner CV could not evaluate any shortlisted hyperparameter settings.")

    summary_rows = sorted(summary_rows, key=lambda r: (-r["mean_inner_val_accuracy"], r["std_inner_val_accuracy"], r["window_size"]))
    best_row = summary_rows[0]
    return best_row, summary_rows, fold_rows, screened_rows


def run_vehicle_experiment(runs_for_vehicle: Sequence[Dict[str, Any]], vehicle_type: str, config: TrainConfig, tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    pilot_ids = get_unique_pilot_ids(runs_for_vehicle)
    if len(pilot_ids) < 3:
        raise ValueError(f"Vehicle {vehicle_type}: need at least three pilots for nested cross-validation.")

    prediction_tables: List[List[Dict[str, Any]]] = []
    outer_fold_rows: List[Dict[str, Any]] = []
    inner_search_rows: List[Dict[str, Any]] = []
    inner_fold_rows: List[Dict[str, Any]] = []
    screening_rows: List[Dict[str, Any]] = []

    candidate_specs = [(int(window_size), tuple(input_vars)) for input_vars in config.input_combinations for window_size in config.window_sizes]
    window_cache = build_window_cache(runs_for_vehicle, candidate_specs, config.stride_fraction, config.max_windows_per_run) if config.precompute_windows else None

    for test_pilot, outer_train_runs, test_runs in get_leave_one_pilot_out_folds(runs_for_vehicle):
        if not outer_train_runs or not test_runs:
            continue
        if len({r["label"] for r in outer_train_runs}) < 2:
            print(f"Skipping {vehicle_type} / {test_pilot}: outer training split has only one class")
            continue

        print(f"\nVehicle {vehicle_type}: outer fold test pilot = {test_pilot}")
        best_row, inner_search, inner_folds, fold_screening = run_inner_cv_hyperparameter_search(
            outer_train_runs,
            vehicle_type,
            config,
            tracker=tracker,
            window_cache=window_cache,
        )
        best_input_vars = tuple(str(best_row["input_vars"]).split(","))
        best_window_size = int(best_row["window_size"])
        selected_num_epochs = max(1, int(round(float(best_row["mean_best_epoch"]))))

        train_windows = collect_windows(outer_train_runs, best_window_size, best_input_vars, config.stride_fraction, config.max_windows_per_run, cache=window_cache)
        test_windows = collect_windows(test_runs, best_window_size, best_input_vars, config.stride_fraction, config.max_windows_per_run, cache=window_cache)
        final_model, _ = fit_lstm_fixed_epochs(
            train_windows,
            config,
            input_size=len(best_input_vars),
            num_epochs=selected_num_epochs,
            tracker=tracker,
            training_label=f"{vehicle_type} | outer-final | test={test_pilot} | vars={','.join(best_input_vars)} | w={best_window_size}",
        )
        test_loader = make_loader(test_windows, config.batch_size, False, config.num_workers, pin_memory=bool(config.pin_memory))
        test_result = evaluate_model(final_model, test_loader, config.device)

        pred_rows = build_prediction_rows(test_result)
        for row in pred_rows:
            row["outer_test_pilot"] = test_pilot
            row["selected_input_vars"] = ",".join(best_input_vars)
            row["selected_window_size"] = best_window_size
            row["selected_num_epochs"] = selected_num_epochs
        prediction_tables.append(pred_rows)

        outer_fold_rows.append({
            "vehicle_type": vehicle_type,
            "outer_test_pilot": test_pilot,
            "selected_input_vars": ",".join(best_input_vars),
            "selected_window_size": best_window_size,
            "selected_num_epochs": selected_num_epochs,
            "num_test_runs": int(len(test_runs)),
            "num_test_windows": int(len(pred_rows)),
            "test_accuracy": float(test_result["accuracy"]),
        })

        for row in inner_search:
            x = dict(row)
            x["outer_test_pilot"] = test_pilot
            inner_search_rows.append(x)
        for row in inner_folds:
            x = dict(row)
            x["outer_test_pilot"] = test_pilot
            inner_fold_rows.append(x)
        for row in fold_screening:
            x = dict(row)
            x["outer_test_pilot"] = test_pilot
            screening_rows.append(x)

    if not prediction_tables:
        raise ValueError(f"No valid outer folds were completed for vehicle {vehicle_type}.")

    combined = combine_prediction_rows(prediction_tables)
    prediction_rows = combined.pop("combined_prediction_rows")
    statistics = run_basic_statistics(prediction_rows)

    return {
        "vehicle_type": vehicle_type,
        "search_rows": inner_search_rows,
        "screening_rows": screening_rows,
        "inner_fold_rows": inner_fold_rows,
        "outer_fold_rows": outer_fold_rows,
        "test_result": combined,
        "prediction_rows": prediction_rows,
        "statistics": statistics,
        "num_outer_folds": int(len(outer_fold_rows)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
        return

    if fieldnames is None:
        keys: List[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_vehicle_results(result: Dict[str, Any], save_dir: str | Path) -> None:
    vehicle_dir = Path(save_dir) / str(result["vehicle_type"])
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    write_csv(vehicle_dir / "screening_scores.csv", result.get("screening_rows", []))
    write_csv(vehicle_dir / "inner_cv_search.csv", result["search_rows"])
    write_csv(vehicle_dir / "inner_cv_fold_scores.csv", result["inner_fold_rows"])
    write_csv(vehicle_dir / "outer_cv_fold_scores.csv", result["outer_fold_rows"])
    write_csv(vehicle_dir / "window_predictions.csv", result["prediction_rows"])
    write_csv(vehicle_dir / "run_level_confidence.csv", result["statistics"].get("aggregated_run_predictions", []))

    cm = result["test_result"]["confusion_matrix"]
    cm_rows = [
        {"": "true_off", "pred_off": int(cm[0, 0]), "pred_on": int(cm[0, 1])},
        {"": "true_on", "pred_off": int(cm[1, 0]), "pred_on": int(cm[1, 1])},
    ]
    write_csv(vehicle_dir / "confusion_matrix.csv", cm_rows, fieldnames=["", "pred_off", "pred_on"])

    summary_payload = {
        "vehicle_type": result["vehicle_type"],
        "num_outer_folds": result["num_outer_folds"],
        "accuracy": result["test_result"]["accuracy"],
        "confusion_matrix": result["test_result"]["confusion_matrix"].tolist(),
        "classification_report": result["test_result"]["classification_report"],
        "statistics": {k: v for k, v in result["statistics"].items() if k != "aggregated_run_predictions"},
    }
    (vehicle_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    with (vehicle_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Vehicle type: {result['vehicle_type']}\n")
        f.write(f"Outer folds (held-out pilots): {result['num_outer_folds']}\n")
        f.write(f"Overall cross-validated accuracy: {result['test_result']['accuracy']:.4f}\n")
        f.write(f"Confusion matrix:\n{result['test_result']['confusion_matrix']}\n\n")
        f.write("Statistics:\n")
        for key, value in result["statistics"].items():
            if key == "aggregated_run_predictions":
                f.write("  aggregated_run_predictions: saved to run_level_confidence.csv\n")
            else:
                f.write(f"  {key}: {value}\n")


def print_dataset_summary(runs: Sequence[Dict[str, Any]]) -> None:
    print("=" * 80)
    print("Dataset summary")
    print(f"Total valid runs: {len(runs)}")
    pilots = sorted({str(r['pilot_id']) for r in runs})
    print(f"Pilots: {', '.join(pilots)}")

    by_vehicle = Counter(str(r["vehicle_type"]) for r in runs)
    by_label = Counter(int(r["label"]) for r in runs)
    print(f"Runs by vehicle: {dict(by_vehicle)}")
    print(f"Runs by label  : {dict(by_label)}")

    for vehicle in sorted(by_vehicle.keys()):
        vruns = [r for r in runs if str(r["vehicle_type"]) == vehicle]
        label_counts = Counter(int(r["label"]) for r in vruns)
        pilot_counts = Counter(str(r["pilot_id"]) for r in vruns)
        print(f"  Vehicle {vehicle}: {len(vruns)} runs, labels={dict(label_counts)}, pilots={dict(pilot_counts)}")
    print("=" * 80)


def print_summary(result: Dict[str, Any]) -> None:
    print("=" * 80)
    print(f"Vehicle type            : {result['vehicle_type']}")
    print(f"Outer CV folds (pilots) : {result['num_outer_folds']}")
    print(f"Overall CV accuracy     : {result['test_result']['accuracy']:.4f}")
    print("Confusion matrix:")
    print(result["test_result"]["confusion_matrix"])
    if result["outer_fold_rows"]:
        print("Outer-fold selections:")
        for row in result["outer_fold_rows"]:
            print(row)
    print("Statistics:")
    for key, value in result["statistics"].items():
        if key == "aggregated_run_predictions":
            print("  aggregated_run_predictions: saved to CSV")
        else:
            print(f"  {key}: {value}")


def main(config: TrainConfig) -> Dict[str, Any]:
    configure_runtime(config)
    set_seed(config.random_state)
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(config, script_dir)
    save_dir = resolve_save_dir(config, script_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data directory: {data_dir}")
    print(f"Saving results to   : {save_dir}")
    print(f"Compute device      : {config.device}")
    if str(config.device).startswith("cuda"):
        print(f"CUDA device         : {torch.cuda.get_device_name(0)}")

    runs = load_runs_from_npz_dir(data_dir, config=config, min_run_length=config.min_run_length)
    print_dataset_summary(runs)

    runs_by_vehicle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_vehicle[str(run["vehicle_type"])].append(run)

    total_trainings = estimate_total_trainings(runs_by_vehicle, config)
    print(f"Estimated total model trainings: {total_trainings}")
    tracker = ProgressTracker(total_trainings=total_trainings)

    all_results: Dict[str, Any] = {}
    for vehicle_type in sorted(runs_by_vehicle.keys()):
        vehicle_runs = runs_by_vehicle[vehicle_type]
        labels = {int(r["label"]) for r in vehicle_runs}
        pilot_ids = get_unique_pilot_ids(vehicle_runs)

        if len(labels) < 2:
            print(f"Skipping {vehicle_type}: only one class present")
            continue
        if len(pilot_ids) < 3:
            print(f"Skipping {vehicle_type}: need at least three pilots")
            continue

        result = run_vehicle_experiment(vehicle_runs, vehicle_type, config, tracker=tracker)
        all_results[vehicle_type] = result
        save_vehicle_results(result, save_dir)
        print_summary(result)

    if not all_results:
        print("No vehicle experiments completed. Check min_run_length and dataset coverage.")
    return all_results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    config = TrainConfig(
        data_dir=Path(r"C:\\Users\\bramb\\Downloads\\AI_project_simulator\\A02-TAS-Repository\\data\\python_data"),
        save_dir=Path(r"C:\\Users\\bramb\\Downloads\\AI_project_simulator\\A02-TAS-Repository\\results_lstm"),
        window_sizes=(64, 96),
        stride_fraction=1.0,
        input_combinations=(
            ("e", "u"),
            ("e", "u", "de", "du"),
        ),
        batch_size=256,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=20,
        patience=4,
        screen_max_epochs=6,
        screen_patience=2,
        screen_num_folds=2,
        top_k_candidates=1,
        max_windows_per_run=24,
        precompute_windows=True,
        random_state=42,
        min_run_length=128,
        num_workers=0,
        device="cuda",
    )
    main(config)
