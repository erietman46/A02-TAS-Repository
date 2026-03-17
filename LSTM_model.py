from __future__ import annotations

"""
LSTM classifier for motion-on vs motion-off behavior in the AE2224-I dataset.

What this script does
---------------------
1. Loads all .npz files from the AE2224-I folder.
2. Normalizes each run separately.
3. Creates short time windows from each run.
4. Trains one LSTM per vehicle type (P, V, A).
5. Uses pilot-wise nested cross-validation:
   - outer loop: train on 5 pilots, test on 1 pilot
   - inner loop: choose the best input signals and window size
6. Uses the GPU automatically when CUDA is available.
7. Saves simple summaries, tables, and graphs that are easy to read.

Folder layout expected by default
---------------------------------
A02-TAS-Repository/
  LSTM_model.py
  data/
    python_data/
      ae2224I_measurement_data_subj1_C1.npz
      ...
      ae2224I_measurement_data_subj6_C6.npz

Condition mapping
-----------------
C1 = Gain (P), no motion
C2 = Single integrator (V), no motion
C3 = Double integrator (A), no motion
C4 = Gain (P), motion
C5 = Single integrator (V), motion
C6 = Double integrator (A), motion
"""

import copy
import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


@dataclass
class TrainConfig:
    data_dir: Optional[str | Path] = None
    save_dir: Optional[str | Path] = None
    window_sizes: Tuple[int, ...] = (32, 64, 96, 128)
    stride_fraction: float = 0.5
    input_combinations: Tuple[Tuple[str, ...], ...] = (
        ("e", "u"),
        ("e", "u", "de"),
        ("e", "u", "du"),
        ("e", "u", "de", "du"),
    )
    batch_size: int = 256
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 7
    random_state: int = 42
    min_run_length: int = 128
    num_workers: int = 0
    device: str = "cuda"
    use_mixed_precision: bool = True
    pin_memory: bool = True
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
        self._print_status("START", newline=True)

    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))
        self._print_status("RUN", newline=False)

    def finish_training(self, actual_epochs: int) -> None:
        self.current_epoch = max(0, int(actual_epochs))
        self.completed_trainings += 1
        self._print_status("DONE", newline=True)

    def _fraction_complete(self) -> float:
        current_fraction = min(1.0, self.current_epoch / max(1, self.current_planned_epochs))
        return min(1.0, (self.completed_trainings + current_fraction) / self.total_trainings)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--:--"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _status_message(self, prefix: str) -> str:
        elapsed = time.time() - self.start_time
        frac = self._fraction_complete()
        eta = (elapsed / frac - elapsed) if frac > 1e-9 else float("inf")
        current_num = min(self.completed_trainings + 1, self.total_trainings)
        return (
            f"[{prefix}] model {current_num}/{self.total_trainings} | "
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
LABEL_NAMES = {0: "No motion", 1: "Motion"}
VEHICLE_NAMES = {"P": "Gain (P)", "V": "Single integrator (V)", "A": "Double integrator (A)"}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_device(config: TrainConfig) -> str:
    requested = str(config.device).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        print("WARNING: CUDA was requested, but no CUDA GPU is available. Falling back to CPU.")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] = (0, 1)) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            cm[idx[yt], idx[yp]] += 1
    return cm


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
            raise ValueError("time must have the same length as the signal")
        return np.gradient(x, time).astype(np.float32)
    dt = 1.0 / fs if fs not in (None, 0) else 1.0
    return np.gradient(x, dt).astype(np.float32)


def normalize_run_signals(run: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(run)
    raw_e = np.asarray(out["e"], dtype=np.float32)
    raw_u = np.asarray(out["u"], dtype=np.float32)
    time_vector = out.get("time")
    fs = out.get("fs")
    out["de"] = safe_zscore(derivative(raw_e, time=time_vector, fs=fs))
    out["du"] = safe_zscore(derivative(raw_u, time=time_vector, fs=fs))
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

    unique_candidates: List[Path] = []
    seen: set[str] = set()
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
        f"{checked}\n\nExpected to find your .npz files in a folder like: data/python_data"
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
        condition = match.group("condition").upper()
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

            if t_key is None:
                t_runs: List[Optional[np.ndarray]] = [None] * len(e_runs)
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
        return self.classifier(h_n[-1])


def collate_fn(batch: Sequence[Any]):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def make_loader(windows: Sequence[Dict[str, Any]], batch_size: int, shuffle: bool, num_workers: int, device: str, pin_memory: bool) -> DataLoader:
    ds = WindowDataset(windows)
    use_pin_memory = bool(pin_memory and device.startswith("cuda"))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        persistent_workers=bool(num_workers > 0),
    )


def make_windows_for_run(run: Dict[str, Any], window_size: int, stride: int, input_vars: Sequence[str]) -> List[Dict[str, Any]]:
    n = len(run["e"])
    rows: List[Dict[str, Any]] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        x = np.stack([run[var][start:end] for var in input_vars], axis=-1)
        rows.append({
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
    return rows


def build_window_table(runs: Sequence[Dict[str, Any]], window_size: int, input_vars: Sequence[str], stride_fraction: float) -> List[Dict[str, Any]]:
    stride = max(1, int(round(window_size * stride_fraction)))
    all_windows: List[Dict[str, Any]] = []
    for run in runs:
        all_windows.extend(make_windows_for_run(run, window_size, stride, input_vars))
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: GradScaler,
    use_mixed_precision: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    use_cuda_amp = bool(use_mixed_precision and device.startswith("cuda"))

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_cuda_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
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
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metas": metas,
    }


def fit_lstm(
    train_windows: Sequence[Dict[str, Any]],
    val_windows: Sequence[Dict[str, Any]],
    config: TrainConfig,
    input_size: int,
    tracker: Optional[ProgressTracker] = None,
    training_label: str = "",
):
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=bool(config.use_mixed_precision and str(config.device).startswith("cuda")))

    train_loader = make_loader(train_windows, config.batch_size, True, config.num_workers, config.device, config.pin_memory)
    val_loader = make_loader(val_windows, config.batch_size, False, config.num_workers, config.device, config.pin_memory)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -float("inf")
    best_epoch = 1
    wait = 0

    if tracker is not None:
        tracker.start_training(training_label or f"train={len(train_windows)} val={len(val_windows)}", config.max_epochs)

    for epoch in range(1, config.max_epochs + 1):
        _ = train_one_epoch(model, train_loader, optimizer, criterion, config.device, scaler, config.use_mixed_precision)
        val_result = evaluate_model(model, val_loader, config.device)
        val_acc = float(val_result["accuracy"])
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
        tracker.finish_training(best_epoch)
    return model, best_epoch, best_val_acc


def fit_lstm_fixed_epochs(
    train_windows: Sequence[Dict[str, Any]],
    config: TrainConfig,
    input_size: int,
    num_epochs: int,
    tracker: Optional[ProgressTracker] = None,
    training_label: str = "",
):
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=bool(config.use_mixed_precision and str(config.device).startswith("cuda")))
    train_loader = make_loader(train_windows, config.batch_size, True, config.num_workers, config.device, config.pin_memory)

    planned_epochs = max(1, int(num_epochs))
    if tracker is not None:
        tracker.start_training(training_label or f"final train={len(train_windows)}", planned_epochs)

    for epoch in range(1, planned_epochs + 1):
        _ = train_one_epoch(model, train_loader, optimizer, criterion, config.device, scaler, config.use_mixed_precision)
        if tracker is not None:
            tracker.update_epoch(epoch)

    if tracker is not None:
        tracker.finish_training(planned_epochs)
    return model


def build_window_prediction_rows(eval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def aggregate_run_predictions(pred_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, int], List[float]] = defaultdict(list)
    for row in pred_rows:
        key = (
            str(row["vehicle_type"]),
            str(row["pilot_id"]),
            str(row["repetition_id"]),
            str(row["run_id"]),
            int(row["true_label"]),
        )
        grouped[key].append(float(row["motion_on_confidence"]))

    run_rows: List[Dict[str, Any]] = []
    for (vehicle_type, pilot_id, repetition_id, run_id, true_label), scores in grouped.items():
        mean_conf = float(np.mean(scores))
        pred_label = int(mean_conf >= 0.5)
        run_rows.append({
            "vehicle_type": vehicle_type,
            "pilot_id": pilot_id,
            "repetition_id": repetition_id,
            "run_id": run_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "mean_motion_on_confidence": mean_conf,
            "num_windows": int(len(scores)),
        })
    run_rows.sort(key=lambda r: (r["vehicle_type"], r["pilot_id"], r["repetition_id"], r["run_id"]))
    return run_rows


def compute_run_metrics(run_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = np.asarray([int(r["true_label"]) for r in run_rows], dtype=int)
    y_pred = np.asarray([int(r["pred_label"]) for r in run_rows], dtype=int)
    cm = confusion_matrix_np(y_true, y_pred, labels=(0, 1))
    correct = int(np.sum(y_true == y_pred))
    total = int(len(y_true))
    recall_off = float(cm[0, 0] / max(1, cm[0, 0] + cm[0, 1]))
    recall_on = float(cm[1, 1] / max(1, cm[1, 0] + cm[1, 1]))
    return {
        "accuracy": float(correct / max(1, total)),
        "correct_runs": correct,
        "total_runs": total,
        "confusion_matrix": cm,
        "recall_no_motion": recall_off,
        "recall_motion": recall_on,
    }


def run_basic_statistics(run_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {"aggregated_run_predictions": list(run_rows)}
    if stats is None:
        results["note"] = "scipy not installed; significance tests skipped"
        return results

    off_scores = np.asarray([r["mean_motion_on_confidence"] for r in run_rows if int(r["true_label"]) == 0], dtype=float)
    on_scores = np.asarray([r["mean_motion_on_confidence"] for r in run_rows if int(r["true_label"]) == 1], dtype=float)
    if len(off_scores) >= 2 and len(on_scores) >= 2:
        t_stat, p_val = stats.ttest_ind(off_scores, on_scores, equal_var=False)
        results["motion_score_difference"] = {
            "mean_no_motion": float(np.mean(off_scores)),
            "mean_motion": float(np.mean(on_scores)),
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
        total += n_pilots * (num_candidates * (n_pilots - 1) + 1)
    return total


def run_inner_cv_hyperparameter_search(
    train_runs: Sequence[Dict[str, Any]],
    vehicle_type: str,
    config: TrainConfig,
    tracker: Optional[ProgressTracker] = None,
):
    inner_folds = get_leave_one_pilot_out_folds(train_runs)
    if len(inner_folds) < 2:
        raise ValueError("Need at least two pilots in the outer-training split for inner cross-validation.")

    summary_rows: List[Dict[str, Any]] = []

    for input_vars in config.input_combinations:
        for window_size in config.window_sizes:
            candidate_rows: List[Dict[str, Any]] = []
            failed = False
            for val_pilot, inner_train_runs, val_runs in inner_folds:
                if not inner_train_runs or not val_runs:
                    failed = True
                    break
                if len({r["label"] for r in inner_train_runs}) < 2:
                    failed = True
                    break
                try:
                    train_windows = build_window_table(inner_train_runs, window_size, input_vars, config.stride_fraction)
                    val_windows = build_window_table(val_runs, window_size, input_vars, config.stride_fraction)
                except ValueError:
                    failed = True
                    break

                _, best_epoch, best_val_acc = fit_lstm(
                    train_windows,
                    val_windows,
                    config,
                    input_size=len(input_vars),
                    tracker=tracker,
                    training_label=f"{vehicle_type} | inner | val={val_pilot} | vars={','.join(input_vars)} | w={window_size}",
                )
                candidate_rows.append({
                    "validation_pilot": val_pilot,
                    "input_vars": ",".join(input_vars),
                    "window_size": int(window_size),
                    "best_epoch": int(best_epoch),
                    "best_val_accuracy": float(best_val_acc),
                })

            if failed or not candidate_rows:
                continue

            accs = [r["best_val_accuracy"] for r in candidate_rows]
            epochs = [r["best_epoch"] for r in candidate_rows]
            summary_rows.append({
                "input_vars": ",".join(input_vars),
                "window_size": int(window_size),
                "mean_inner_val_accuracy": float(np.mean(accs)),
                "std_inner_val_accuracy": float(np.std(accs, ddof=0)),
                "mean_best_epoch": float(np.mean(epochs)),
                "num_inner_folds": int(len(candidate_rows)),
            })

    if not summary_rows:
        raise ValueError("Inner cross-validation could not evaluate any hyperparameter settings.")

    summary_rows.sort(key=lambda r: (-r["mean_inner_val_accuracy"], r["std_inner_val_accuracy"], r["window_size"]))
    return summary_rows[0]


def run_vehicle_experiment(
    runs_for_vehicle: Sequence[Dict[str, Any]],
    vehicle_type: str,
    config: TrainConfig,
    tracker: Optional[ProgressTracker] = None,
) -> Dict[str, Any]:
    pilot_ids = get_unique_pilot_ids(runs_for_vehicle)
    if len(pilot_ids) < 3:
        raise ValueError(f"Vehicle {vehicle_type}: need at least three pilots for nested cross-validation.")

    all_window_pred_rows: List[Dict[str, Any]] = []
    outer_fold_rows: List[Dict[str, Any]] = []

    for test_pilot, outer_train_runs, test_runs in get_leave_one_pilot_out_folds(runs_for_vehicle):
        if not outer_train_runs or not test_runs:
            continue
        if len({r["label"] for r in outer_train_runs}) < 2:
            print(f"Skipping {vehicle_type} / {test_pilot}: outer training split has only one class")
            continue

        print(f"\nVehicle {vehicle_type}: held-out test pilot = {test_pilot}")
        best = run_inner_cv_hyperparameter_search(outer_train_runs, vehicle_type, config, tracker)
        best_input_vars = tuple(str(best["input_vars"]).split(","))
        best_window_size = int(best["window_size"])
        selected_num_epochs = max(1, int(round(float(best["mean_best_epoch"]))))

        train_windows = build_window_table(outer_train_runs, best_window_size, best_input_vars, config.stride_fraction)
        test_windows = build_window_table(test_runs, best_window_size, best_input_vars, config.stride_fraction)

        final_model = fit_lstm_fixed_epochs(
            train_windows,
            config,
            input_size=len(best_input_vars),
            num_epochs=selected_num_epochs,
            tracker=tracker,
            training_label=f"{vehicle_type} | final | test={test_pilot} | vars={','.join(best_input_vars)} | w={best_window_size}",
        )
        test_loader = make_loader(test_windows, config.batch_size, False, config.num_workers, config.device, config.pin_memory)
        test_result = evaluate_model(final_model, test_loader, config.device)
        fold_window_rows = build_window_prediction_rows(test_result)
        fold_run_rows = aggregate_run_predictions(fold_window_rows)
        fold_metrics = compute_run_metrics(fold_run_rows)

        for row in fold_window_rows:
            row["outer_test_pilot"] = test_pilot
            row["selected_input_vars"] = ",".join(best_input_vars)
            row["selected_window_size"] = best_window_size
            row["selected_num_epochs"] = selected_num_epochs
            all_window_pred_rows.append(row)

        outer_fold_rows.append({
            "held_out_pilot": test_pilot,
            "best_inputs": ", ".join(best_input_vars),
            "best_window_size": best_window_size,
            "training_epochs": selected_num_epochs,
            "correct_runs": int(fold_metrics["correct_runs"]),
            "total_runs": int(fold_metrics["total_runs"]),
            "run_accuracy": float(fold_metrics["accuracy"]),
        })

    if not all_window_pred_rows:
        raise ValueError(f"No valid outer folds were completed for vehicle {vehicle_type}.")

    run_rows = aggregate_run_predictions(all_window_pred_rows)
    run_metrics = compute_run_metrics(run_rows)
    statistics = run_basic_statistics(run_rows)

    return {
        "vehicle_type": vehicle_type,
        "vehicle_name": VEHICLE_NAMES.get(vehicle_type, vehicle_type),
        "outer_fold_rows": outer_fold_rows,
        "run_rows": run_rows,
        "window_rows": all_window_pred_rows,
        "summary_metrics": run_metrics,
        "statistics": statistics,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            if fieldnames is not None:
                writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                writer.writeheader()
        return

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def format_simple_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    widths = [len(h) for h in headers]
    data_rows: List[List[str]] = []
    for row in rows:
        rendered = [str(row.get(key, "")) for key, _ in columns]
        data_rows.append(rendered)
        widths = [max(w, len(cell)) for w, cell in zip(widths, rendered)]

    def render_line(values: Sequence[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    sep = "-+-".join("-" * w for w in widths)
    out = [render_line(headers), sep]
    out.extend(render_line(r) for r in data_rows)
    return "\n".join(out)


def print_dataset_summary(runs: Sequence[Dict[str, Any]]) -> None:
    total_runs = len(runs)
    pilots = sorted({str(r["pilot_id"]) for r in runs})
    by_vehicle = Counter(str(r["vehicle_type"]) for r in runs)
    print("=" * 88)
    print("DATASET OVERVIEW")
    print(f"Valid runs : {total_runs}")
    print(f"Pilots     : {', '.join(pilots)}")
    print(f"Vehicles   : {', '.join(f'{VEHICLE_NAMES.get(v, v)} ({by_vehicle[v]} runs)' for v in sorted(by_vehicle))}")

    rows = []
    for vehicle in sorted(by_vehicle):
        vruns = [r for r in runs if str(r["vehicle_type"]) == vehicle]
        label_counts = Counter(int(r["label"]) for r in vruns)
        rows.append({
            "vehicle": VEHICLE_NAMES.get(vehicle, vehicle),
            "runs": len(vruns),
            "no_motion": int(label_counts.get(0, 0)),
            "motion": int(label_counts.get(1, 0)),
        })
    print(format_simple_table(rows, [("vehicle", "Vehicle"), ("runs", "Runs"), ("no_motion", "No motion"), ("motion", "Motion")]))
    print("=" * 88)


def print_vehicle_summary(result: Dict[str, Any]) -> None:
    metrics = result["summary_metrics"]
    cm = metrics["confusion_matrix"]
    print("=" * 88)
    print(f"RESULTS FOR {result['vehicle_name']}")
    print(f"Run-level accuracy: {format_pct(metrics['accuracy'])} ({metrics['correct_runs']}/{metrics['total_runs']} correct runs)")
    print(f"No-motion detection: {format_pct(metrics['recall_no_motion'])}")
    print(f"Motion detection   : {format_pct(metrics['recall_motion'])}")
    print("Confusion matrix (run level)")
    print(format_simple_table([
        {"true": "No motion", "pred_no": int(cm[0, 0]), "pred_yes": int(cm[0, 1])},
        {"true": "Motion", "pred_no": int(cm[1, 0]), "pred_yes": int(cm[1, 1])},
    ], [("true", "Actual"), ("pred_no", "Predicted no motion"), ("pred_yes", "Predicted motion")]))

    fold_rows = []
    for row in result["outer_fold_rows"]:
        fold_rows.append({
            "pilot": row["held_out_pilot"],
            "accuracy": format_pct(row["run_accuracy"]),
            "inputs": row["best_inputs"],
            "window": row["best_window_size"],
        })
    print("Held-out pilot results")
    print(format_simple_table(fold_rows, [("pilot", "Pilot"), ("accuracy", "Accuracy"), ("inputs", "Best inputs"), ("window", "Window")]))

    stat = result["statistics"].get("motion_score_difference")
    if stat is not None:
        print(
            f"Average motion score: no motion = {stat['mean_no_motion']:.3f}, motion = {stat['mean_motion']:.3f}, "
            f"p-value = {stat['p_value']:.4g}"
        )
    print("=" * 88)


def save_confusion_matrix_plot(cm: np.ndarray, title: str, path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], ["Predicted\nNo motion", "Predicted\nMotion"])
    ax.set_yticks([0, 1], ["Actual\nNo motion", "Actual\nMotion"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fold_accuracy_plot(rows: Sequence[Dict[str, Any]], title: str, path: Path) -> None:
    if plt is None or not rows:
        return
    labels = [str(r["held_out_pilot"]) for r in rows]
    values = [100.0 * float(r["run_accuracy"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_motion_score_plot(run_rows: Sequence[Dict[str, Any]], title: str, path: Path) -> None:
    if plt is None or not run_rows:
        return
    no_motion = [float(r["mean_motion_on_confidence"]) for r in run_rows if int(r["true_label"]) == 0]
    motion = [float(r["mean_motion_on_confidence"]) for r in run_rows if int(r["true_label"]) == 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([no_motion, motion], labels=["Actual no motion", "Actual motion"])
    ax.set_ylabel("Average motion score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overall_accuracy_plot(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if plt is None or not rows:
        return
    labels = [str(r["vehicle_name"]) for r in rows]
    values = [100.0 * float(r["run_accuracy"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final accuracy by vehicle type")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_vehicle_results(result: Dict[str, Any], save_dir: str | Path) -> None:
    vehicle_dir = Path(save_dir) / str(result["vehicle_type"])
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    run_rows_to_save = []
    for row in result["run_rows"]:
        x = dict(row)
        x["true_class"] = LABEL_NAMES[int(x["true_label"])]
        x["predicted_class"] = LABEL_NAMES[int(x["pred_label"])]
        run_rows_to_save.append(x)

    write_csv(vehicle_dir / "pilot_results.csv", result["outer_fold_rows"])
    write_csv(vehicle_dir / "run_predictions.csv", run_rows_to_save)

    cm = result["summary_metrics"]["confusion_matrix"]
    cm_rows = [
        {"actual": "No motion", "predicted_no_motion": int(cm[0, 0]), "predicted_motion": int(cm[0, 1])},
        {"actual": "Motion", "predicted_no_motion": int(cm[1, 0]), "predicted_motion": int(cm[1, 1])},
    ]
    write_csv(vehicle_dir / "confusion_matrix.csv", cm_rows)

    save_confusion_matrix_plot(cm, f"{result['vehicle_name']} - confusion matrix", vehicle_dir / "confusion_matrix.png")
    save_fold_accuracy_plot(result["outer_fold_rows"], f"{result['vehicle_name']} - held-out pilot accuracy", vehicle_dir / "pilot_accuracy.png")
    save_motion_score_plot(result["run_rows"], f"{result['vehicle_name']} - average motion score per run", vehicle_dir / "motion_score_boxplot.png")

    metrics = result["summary_metrics"]
    summary_payload = {
        "vehicle_type": result["vehicle_type"],
        "vehicle_name": result["vehicle_name"],
        "run_accuracy": metrics["accuracy"],
        "correct_runs": metrics["correct_runs"],
        "total_runs": metrics["total_runs"],
        "recall_no_motion": metrics["recall_no_motion"],
        "recall_motion": metrics["recall_motion"],
        "statistics": {k: v for k, v in result["statistics"].items() if k != "aggregated_run_predictions"},
    }
    (vehicle_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    stat = result["statistics"].get("motion_score_difference")
    with (vehicle_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Vehicle type: {result['vehicle_name']}\n")
        f.write(f"Run-level accuracy: {format_pct(metrics['accuracy'])} ({metrics['correct_runs']}/{metrics['total_runs']} correct runs)\n")
        f.write(f"No-motion detection: {format_pct(metrics['recall_no_motion'])}\n")
        f.write(f"Motion detection: {format_pct(metrics['recall_motion'])}\n")
        if stat is not None:
            f.write(
                f"Average motion score: no motion = {stat['mean_no_motion']:.3f}, "
                f"motion = {stat['mean_motion']:.3f}, p-value = {stat['p_value']:.4g}\n"
            )
        f.write("\nFiles in this folder:\n")
        f.write("- summary.txt: short explanation\n")
        f.write("- pilot_results.csv: accuracy per held-out pilot\n")
        f.write("- run_predictions.csv: one row per run\n")
        f.write("- confusion_matrix.csv and confusion_matrix.png\n")
        f.write("- pilot_accuracy.png\n")
        f.write("- motion_score_boxplot.png\n")


def save_overall_results(all_results: Dict[str, Any], save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    rows = []
    for vehicle_type in sorted(all_results):
        result = all_results[vehicle_type]
        metrics = result["summary_metrics"]
        rows.append({
            "vehicle_type": vehicle_type,
            "vehicle_name": result["vehicle_name"],
            "run_accuracy": float(metrics["accuracy"]),
            "correct_runs": int(metrics["correct_runs"]),
            "total_runs": int(metrics["total_runs"]),
            "recall_no_motion": float(metrics["recall_no_motion"]),
            "recall_motion": float(metrics["recall_motion"]),
        })
    write_csv(save_dir / "overall_results.csv", rows)
    save_overall_accuracy_plot(rows, save_dir / "overall_accuracy.png")

    human_lines = ["OVERALL RESULTS", ""]
    for row in rows:
        human_lines.append(
            f"{row['vehicle_name']}: accuracy {format_pct(row['run_accuracy'])} "
            f"({row['correct_runs']}/{row['total_runs']} correct runs)"
        )
    (save_dir / "overall_results.txt").write_text("\n".join(human_lines), encoding="utf-8")


def main(config: TrainConfig) -> Dict[str, Any]:
    set_seed(config.random_state)
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(config, script_dir)
    save_dir = resolve_save_dir(config, script_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config.device = resolve_device(config)
    print(f"Using data directory : {data_dir}")
    print(f"Saving results to    : {save_dir}")
    print(f"Using compute device : {config.device}")
    if config.device.startswith("cuda"):
        print(f"GPU name             : {torch.cuda.get_device_name(0)}")
    print()

    runs = load_runs_from_npz_dir(data_dir, config=config, min_run_length=config.min_run_length)
    print_dataset_summary(runs)

    runs_by_vehicle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_vehicle[str(run["vehicle_type"])].append(run)

    total_trainings = estimate_total_trainings(runs_by_vehicle, config)
    print(f"Estimated total model fits: {total_trainings}")
    tracker = ProgressTracker(total_trainings=total_trainings)

    all_results: Dict[str, Any] = {}
    for vehicle_type in sorted(runs_by_vehicle):
        vehicle_runs = runs_by_vehicle[vehicle_type]
        labels = {int(r["label"]) for r in vehicle_runs}
        pilot_ids = get_unique_pilot_ids(vehicle_runs)
        if len(labels) < 2:
            print(f"Skipping {vehicle_type}: only one class is present")
            continue
        if len(pilot_ids) < 3:
            print(f"Skipping {vehicle_type}: at least three pilots are required")
            continue

        result = run_vehicle_experiment(vehicle_runs, vehicle_type, config, tracker)
        all_results[vehicle_type] = result
        save_vehicle_results(result, save_dir)
        print_vehicle_summary(result)

    if not all_results:
        print("No vehicle experiments completed. Check min_run_length and dataset coverage.")
        return {}

    save_overall_results(all_results, save_dir)
    print(f"All summary files saved in: {save_dir}")
    return all_results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    config = TrainConfig(
        data_dir=base_dir / "data" / "python_data",
        save_dir=base_dir / "results_lstm",
        window_sizes=(32, 64, 96, 128),
        stride_fraction=0.5,
        input_combinations=(
            ("e", "u"),
            ("e", "u", "de"),
            ("e", "u", "du"),
            ("e", "u", "de", "du"),
        ),
        batch_size=256,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=40,
        patience=7,
        random_state=42,
        min_run_length=128,
        num_workers=0,
        device="cuda",
        use_mixed_precision=True,
        pin_memory=True,
    )
    main(config)
