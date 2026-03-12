"""
LSTM-based time-series classifier for motion-on vs motion-off behavior.

Pipeline implemented:
1. Load run-level data from .npz files that match the AE2224-I naming convention.
2. Normalize each run separately.
3. Build labeled short windows from each run.
4. Train an LSTM classifier per vehicle type.
5. Tune window size and input-variable combination.
6. Evaluate with accuracy and confusion matrix.
7. Compare confidence scores statistically at run level.

Expected signal keys inside each .npz file:
    e, u, and optionally t

Each file can contain multiple runs, for example MATLAB-style object arrays of shape (20, 1).
Metadata such as pilot, condition, repetition, label, and vehicle type are derived from the filename and configuration mappings.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    data_dir: str = "python_data"
    window_sizes: Tuple[int, ...] = (32, 64, 96, 128)
    stride_fraction: float = 0.5
    input_combinations: Tuple[Tuple[str, ...], ...] = (
        ("e", "u"),
        ("e", "u", "de"),
        ("e", "u", "du"),
        ("e", "u", "de", "du"),
    )
    batch_size: int = 128
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results_lstm"
    filename_pattern: str = r"^ae2224I_measurement_data_subj(?P<subject>[1-6])_(?P<condition>C[1-6])\.npz$"
    signal_keys: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "e": ("e",),
        "u": ("u",),
        "time": ("t", "time"),
    })
    condition_to_label: Dict[str, int] = field(default_factory=lambda: {
        "C1": 0,  # Gain (P), no motion
        "C2": 0,  # Single integrator (V), no motion
        "C3": 0,  # Double integrator (A), no motion
        "C4": 1,  # Gain (P), motion
        "C5": 1,  # Single integrator (V), motion
        "C6": 1,  # Double integrator (A), motion
    })
    condition_to_vehicle: Dict[str, str] = field(default_factory=lambda: {
        "C1": "P",
        "C2": "V",
        "C3": "A",
        "C4": "P",
        "C5": "V",
        "C6": "A",
    })


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        return x - mu
    return (x - mu) / sigma


def derivative(x: np.ndarray, time: np.ndarray | None = None, fs: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if time is not None:
        return np.gradient(x, np.asarray(time, dtype=np.float32)).astype(np.float32)
    dt = 1.0 / fs if fs not in (None, 0) else 1.0
    return np.gradient(x, dt).astype(np.float32)


def normalize_run_signals(run: Dict) -> Dict:
    run = copy.deepcopy(run)
    run["e"] = safe_zscore(run["e"])
    run["u"] = safe_zscore(run["u"])
    time = run.get("time")
    fs = run.get("fs")
    run["de"] = safe_zscore(derivative(run["e"], time=time, fs=fs))
    run["du"] = safe_zscore(derivative(run["u"], time=time, fs=fs))
    return run

FILENAME_HELP = "ae2224I_measurement_data_subj<1-6>_C<1-6>.npz"


def _first_available_key(npz_obj, candidates: Sequence[str], *, required: bool = True):
    for key in candidates:
        if key in npz_obj.files:
            return key
    if required:
        raise KeyError(f"None of the keys {candidates} were found in {npz_obj.files}")
    return None


def _to_1d_float_array(value) -> np.ndarray:
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.reshape(-1)[0])
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _split_npz_field_into_runs(value) -> List[np.ndarray]:
    arr = np.asarray(value, allow_pickle=True)
    if arr.dtype == object:
        return [_to_1d_float_array(item) for item in arr.reshape(-1)]
    arr = np.asarray(arr)
    if arr.ndim <= 1:
        return [_to_1d_float_array(arr)]
    if arr.ndim == 2 and 1 in arr.shape:
        flat = arr.reshape(-1)
        if flat.dtype == object:
            return [_to_1d_float_array(item) for item in flat]
        return [_to_1d_float_array(arr)]
    return [_to_1d_float_array(arr)]


def _infer_fs_from_time(time_vector: np.ndarray) -> float | None:
    if time_vector is None or len(time_vector) < 2:
        return None
    dt = np.diff(time_vector.astype(np.float32))
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return None
    median_dt = float(np.median(dt))
    if abs(median_dt) < 1e-12:
        return None
    return float(1.0 / median_dt)


def _condition_label(condition: str, config: TrainConfig) -> int:
    if condition in config.condition_to_label:
        return int(config.condition_to_label[condition])
    raise ValueError(
        f"No label mapping configured for {condition}. "
        f"Set TrainConfig.condition_to_label, for example {{'C1': 0, 'C2': 1, ...}}."
    )


def _condition_vehicle(condition: str, config: TrainConfig) -> str:
    return str(config.condition_to_vehicle.get(condition, condition))


def load_runs_from_npz_dir(data_dir: str, config: TrainConfig, min_run_length: int = 1) -> List[Dict]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pattern = re.compile(config.filename_pattern, re.IGNORECASE)
    npz_files = [p for p in sorted(data_path.glob('*.npz')) if pattern.match(p.name)]
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files matched the expected naming pattern in {data_dir}. Expected format: {FILENAME_HELP}"
        )

    expected = {f"ae2224I_measurement_data_subj{s}_C{c}.npz".lower() for s in range(1, 7) for c in range(1, 7)}
    found = {p.name.lower() for p in npz_files}
    missing = sorted(expected - found)
    if missing:
        print("Warning: missing expected files:")
        for name in missing:
            print(f"  - {name}")

    all_runs: List[Dict] = []
    for npz_file in npz_files:
        match = pattern.match(npz_file.name)
        if match is None:
            continue
        subject = f"subj{match.group('subject')}"
        condition = match.group('condition').upper()

        with np.load(npz_file, allow_pickle=True) as npz_obj:
            e_key = _first_available_key(npz_obj, config.signal_keys['e'])
            u_key = _first_available_key(npz_obj, config.signal_keys['u'])
            t_key = _first_available_key(npz_obj, config.signal_keys['time'], required=False)

            e_runs = _split_npz_field_into_runs(npz_obj[e_key])
            u_runs = _split_npz_field_into_runs(npz_obj[u_key])
            t_runs = _split_npz_field_into_runs(npz_obj[t_key]) if t_key is not None else [None] * len(e_runs)

            if len(u_runs) != len(e_runs):
                raise ValueError(f"{npz_file.name}: e has {len(e_runs)} runs but u has {len(u_runs)} runs.")
            if t_key is not None and len(t_runs) != len(e_runs):
                raise ValueError(f"{npz_file.name}: e has {len(e_runs)} runs but t has {len(t_runs)} runs.")

            for idx, (e_run, u_run) in enumerate(zip(e_runs, u_runs), start=1):
                time_run = t_runs[idx - 1] if idx - 1 < len(t_runs) else None
                n = min(len(e_run), len(u_run), len(time_run) if time_run is not None else 10**12)
                if n < min_run_length:
                    continue

                cleaned = {
                    'e': np.asarray(e_run[:n], dtype=np.float32),
                    'u': np.asarray(u_run[:n], dtype=np.float32),
                    'label': _condition_label(condition, config),
                    'vehicle_type': _condition_vehicle(condition, config),
                    'pilot_id': subject,
                    'condition_id': condition,
                    'repetition_id': f'rep{idx}',
                    'source_file': npz_file.name,
                    'run_id': f'{npz_file.stem}_rep{idx}',
                }
                if time_run is not None:
                    cleaned['time'] = np.asarray(time_run[:n], dtype=np.float32)
                    fs = _infer_fs_from_time(cleaned['time'])
                    if fs is not None:
                        cleaned['fs'] = fs

                all_runs.append(normalize_run_signals(cleaned))

    if not all_runs:
        raise ValueError('No valid runs were loaded from the .npz files. Check run lengths and naming.')
    return all_runs


def extract_runs_from_module(module) -> List[Dict]:
    if hasattr(module, "get_runs") and callable(module.get_runs):
        runs = module.get_runs()
    elif hasattr(module, "RUNS"):
        runs = module.RUNS
    elif hasattr(module, "DATA"):
        runs = module.DATA
    else:
        raise ValueError(
            f"Could not find runs in module {module.__name__}. Expected get_runs(), RUNS, or DATA."
        )
    if not isinstance(runs, (list, tuple)):
        raise TypeError(f"Runs in module {module.__name__} must be a list or tuple.")
    return list(runs)


def import_python_module(py_file: Path):
    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runs_from_python_dir(data_dir: str, min_run_length: int = 1) -> List[Dict]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    py_files = [
        p for p in data_path.glob("*.py")
        if p.name != Path(__file__).name and not p.name.startswith("__")
    ]
    if not py_files:
        raise FileNotFoundError(f"No Python data files found in: {data_dir}")

    all_runs = []
    for py_file in sorted(py_files):
        module = import_python_module(py_file)
        runs = extract_runs_from_module(module)

        for idx, run in enumerate(runs):
            required = {"e", "u", "label", "vehicle_type", "pilot_id", "repetition_id"}
            missing = required - set(run.keys())
            if missing:
                raise KeyError(f"{py_file.name}, run {idx}: missing keys {missing}")

            n = min(len(run["e"]), len(run["u"]))
            if n < min_run_length:
                continue

            cleaned = {
                "e": np.asarray(run["e"][:n], dtype=np.float32),
                "u": np.asarray(run["u"][:n], dtype=np.float32),
                "label": int(run["label"]),
                "vehicle_type": str(run["vehicle_type"]),
                "pilot_id": str(run["pilot_id"]),
                "repetition_id": str(run["repetition_id"]),
                "source_file": py_file.name,
                "run_id": f"{py_file.stem}_run{idx}",
            }
            if "time" in run and run["time"] is not None:
                cleaned["time"] = np.asarray(run["time"][:n], dtype=np.float32)
            if "fs" in run and run["fs"] is not None:
                cleaned["fs"] = float(run["fs"])

            all_runs.append(normalize_run_signals(cleaned))

    if not all_runs:
        raise ValueError("No valid runs were loaded.")
    return all_runs


def make_windows_for_run(run: Dict, window_size: int, stride: int, input_vars: Sequence[str]) -> List[Dict]:
    n = len(run["e"])
    windows = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        x = np.stack([run[var][start:end] for var in input_vars], axis=-1)
        windows.append(
            {
                "x": x.astype(np.float32),
                "y": int(run["label"]),
                "vehicle_type": run["vehicle_type"],
                "pilot_id": run["pilot_id"],
                "repetition_id": run["repetition_id"],
                "run_id": run["run_id"],
                "source_file": run["source_file"],
                "start_idx": start,
                "end_idx": end,
            }
        )
    return windows


def build_window_table(runs: Sequence[Dict], window_size: int, input_vars: Sequence[str], stride_fraction: float) -> List[Dict]:
    stride = max(1, int(window_size * stride_fraction))
    all_windows = []
    for run in runs:
        all_windows.extend(make_windows_for_run(run, window_size, stride, input_vars))
    if not all_windows:
        raise ValueError("No windows could be created. Check run length and window size.")
    return all_windows


class WindowDataset(Dataset):
    def __init__(self, windows: Sequence[Dict]):
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
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, num_classes: int = 2):
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

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)


def collate_fn(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def make_loader(windows: Sequence[Dict], batch_size: int, shuffle: bool, num_workers: int):
    ds = WindowDataset(windows)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


def get_unique_pilot_ids(runs: Sequence[Dict]) -> List[str]:
    return sorted({str(r["pilot_id"]) for r in runs})


def split_runs_by_pilot(runs: Sequence[Dict], held_out_pilot: str) -> Tuple[List[Dict], List[Dict]]:
    train_runs = [r for r in runs if str(r["pilot_id"]) != str(held_out_pilot)]
    test_runs = [r for r in runs if str(r["pilot_id"]) == str(held_out_pilot)]
    return train_runs, test_runs


def get_leave_one_pilot_out_folds(runs: Sequence[Dict]) -> List[Tuple[str, List[Dict], List[Dict]]]:
    folds = []
    for pilot_id in get_unique_pilot_ids(runs):
        train_runs, test_runs = split_runs_by_pilot(runs, pilot_id)
        folds.append((pilot_id, train_runs, test_runs))
    return folds


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob, metas = [], [], [], []
    for x, y, meta in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs[:, 1].tolist())
        metas.extend(meta)
    return np.array(y_true), np.array(y_pred), np.array(y_prob), metas


def evaluate_model(model, loader, device):
    y_true, y_pred, y_prob, metas = predict_loader(model, loader, device)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metas": metas,
    }


def fit_lstm(train_windows: Sequence[Dict], val_windows: Sequence[Dict], config: TrainConfig, input_size: int):
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(train_windows, config.batch_size, True, config.num_workers)
    val_loader = make_loader(val_windows, config.batch_size, False, config.num_workers)

    best_model = None
    best_val_acc = -np.inf
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_result = evaluate_model(model, val_loader, config.device)
        val_acc = val_result["accuracy"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    model.load_state_dict(best_model)
    return model, pd.DataFrame(history), best_epoch, best_val_acc


def build_prediction_table(eval_result: Dict) -> pd.DataFrame:
    rows = []
    for yt, yp, pr, meta in zip(eval_result["y_true"], eval_result["y_pred"], eval_result["y_prob"], eval_result["metas"]):
        rows.append(
            {
                "true_label": int(yt),
                "pred_label": int(yp),
                "motion_on_confidence": float(pr),
                "vehicle_type": meta["vehicle_type"],
                "pilot_id": meta["pilot_id"],
                "repetition_id": meta["repetition_id"],
                "run_id": meta["run_id"],
                "source_file": meta["source_file"],
                "start_idx": meta["start_idx"],
                "end_idx": meta["end_idx"],
            }
        )
    return pd.DataFrame(rows)


def combine_prediction_tables(prediction_tables: Sequence[pd.DataFrame]) -> Dict:
    if not prediction_tables:
        raise ValueError("No prediction tables were provided for combination.")

    combined = pd.concat(prediction_tables, ignore_index=True)
    y_true = combined["true_label"].to_numpy(dtype=int)
    y_pred = combined["pred_label"].to_numpy(dtype=int)
    y_prob = combined["motion_on_confidence"].to_numpy(dtype=float)

    return {
        "combined_prediction_df": combined,
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def fit_lstm_fixed_epochs(train_windows: Sequence[Dict], config: TrainConfig, input_size: int, num_epochs: int):
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_windows, config.batch_size, True, config.num_workers)

    history = []
    for epoch in range(1, max(1, int(num_epochs)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        history.append({"epoch": epoch, "train_loss": train_loss})

    return model, pd.DataFrame(history)


def run_inner_cv_hyperparameter_search(train_runs: Sequence[Dict], vehicle_type: str, config: TrainConfig):
    inner_folds = get_leave_one_pilot_out_folds(train_runs)
    if len(inner_folds) < 2:
        raise ValueError("At least two pilots are required in the training split for inner cross-validation.")

    summary_rows = []
    fold_rows = []

    for input_vars in config.input_combinations:
        for window_size in config.window_sizes:
            fold_metrics = []
            candidate_failed = False

            for val_pilot, inner_train_runs, val_runs in inner_folds:
                if len(inner_train_runs) == 0 or len(val_runs) == 0:
                    candidate_failed = True
                    break
                if len({r["label"] for r in inner_train_runs}) < 2:
                    candidate_failed = True
                    break

                try:
                    train_windows = build_window_table(inner_train_runs, window_size, input_vars, config.stride_fraction)
                    val_windows = build_window_table(val_runs, window_size, input_vars, config.stride_fraction)
                except ValueError:
                    candidate_failed = True
                    break

                model, _, best_epoch, best_val_acc = fit_lstm(
                    train_windows=train_windows,
                    val_windows=val_windows,
                    config=config,
                    input_size=len(input_vars),
                )

                fold_metrics.append(
                    {
                        "vehicle_type": vehicle_type,
                        "validation_pilot": val_pilot,
                        "input_vars": ",".join(input_vars),
                        "window_size": int(window_size),
                        "best_epoch": int(best_epoch),
                        "best_val_accuracy": float(best_val_acc),
                    }
                )

            if candidate_failed or not fold_metrics:
                continue

            fold_df = pd.DataFrame(fold_metrics)
            fold_rows.append(fold_df)
            summary_rows.append(
                {
                    "vehicle_type": vehicle_type,
                    "input_vars": ",".join(input_vars),
                    "window_size": int(window_size),
                    "mean_inner_val_accuracy": float(fold_df["best_val_accuracy"].mean()),
                    "std_inner_val_accuracy": float(fold_df["best_val_accuracy"].std(ddof=0)),
                    "mean_best_epoch": float(fold_df["best_epoch"].mean()),
                    "num_inner_folds": int(len(fold_df)),
                }
            )

    if not summary_rows:
        raise ValueError("Inner cross-validation could not evaluate any hyperparameter setting.")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_inner_val_accuracy", "std_inner_val_accuracy"],
        ascending=[False, True],
    ).reset_index(drop=True)
    fold_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    best_row = summary_df.iloc[0].to_dict()
    return best_row, summary_df, fold_df


def run_basic_statistics(pred_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Confidence values are aggregated per run before inferential testing,
    because overlapping windows are not independent observations.
    """
    results: Dict[str, Dict] = {}

    run_df = (
        pred_df.groupby(["vehicle_type", "pilot_id", "repetition_id", "run_id", "true_label"], as_index=False)["motion_on_confidence"]
        .mean()
        .rename(columns={"motion_on_confidence": "mean_motion_on_confidence"})
    )

    off_scores = run_df.loc[run_df["true_label"] == 0, "mean_motion_on_confidence"].values
    on_scores = run_df.loc[run_df["true_label"] == 1, "mean_motion_on_confidence"].values

    if len(off_scores) >= 2 and len(on_scores) >= 2:
        t_stat, p_val = stats.ttest_ind(off_scores, on_scores, equal_var=False)
        results["t_test_confidence_off_vs_on"] = {
            "n_off_runs": int(len(off_scores)),
            "n_on_runs": int(len(on_scores)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "mean_off": float(np.mean(off_scores)),
            "mean_on": float(np.mean(on_scores)),
        }

    pilot_groups = [grp["mean_motion_on_confidence"].values for _, grp in run_df.groupby("pilot_id") if len(grp) >= 2]
    if len(pilot_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*pilot_groups)
        results["anova_by_pilot"] = {
            "num_groups": len(pilot_groups),
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
        }

    repetition_groups = [grp["mean_motion_on_confidence"].values for _, grp in run_df.groupby("repetition_id") if len(grp) >= 2]
    if len(repetition_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*repetition_groups)
        results["anova_by_repetition"] = {
            "num_groups": len(repetition_groups),
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
        }

    results["aggregated_run_predictions"] = run_df
    return results


def run_vehicle_experiment(runs_for_vehicle: Sequence[Dict], vehicle_type: str, config: TrainConfig) -> Dict:
    pilot_ids = get_unique_pilot_ids(runs_for_vehicle)
    if len(pilot_ids) < 3:
        raise ValueError("At least three pilots are required for nested pilot-wise cross-validation.")

    prediction_tables = []
    outer_fold_rows = []
    inner_search_tables = []
    inner_fold_tables = []

    for test_pilot, outer_train_runs, test_runs in get_leave_one_pilot_out_folds(runs_for_vehicle):
        if len(outer_train_runs) == 0 or len(test_runs) == 0:
            continue
        if len({r["label"] for r in outer_train_runs}) < 2:
            print(f"Skipping outer fold {vehicle_type} / {test_pilot}: training split has only one class.")
            continue

        best_row, inner_search_df, inner_fold_df = run_inner_cv_hyperparameter_search(
            outer_train_runs,
            vehicle_type,
            config,
        )
        best_input_vars = tuple(str(best_row["input_vars"]).split(","))
        best_window_size = int(best_row["window_size"])
        selected_num_epochs = max(1, int(round(float(best_row["mean_best_epoch"]))))

        train_windows = build_window_table(outer_train_runs, best_window_size, best_input_vars, config.stride_fraction)
        test_windows = build_window_table(test_runs, best_window_size, best_input_vars, config.stride_fraction)

        final_model, _ = fit_lstm_fixed_epochs(
            train_windows=train_windows,
            config=config,
            input_size=len(best_input_vars),
            num_epochs=selected_num_epochs,
        )

        test_loader = make_loader(test_windows, config.batch_size, False, config.num_workers)
        test_result = evaluate_model(final_model, test_loader, config.device)
        pred_df = build_prediction_table(test_result)
        pred_df["outer_test_pilot"] = test_pilot
        pred_df["selected_input_vars"] = ",".join(best_input_vars)
        pred_df["selected_window_size"] = best_window_size
        pred_df["selected_num_epochs"] = selected_num_epochs
        prediction_tables.append(pred_df)

        fold_accuracy = accuracy_score(pred_df["true_label"], pred_df["pred_label"])
        outer_fold_rows.append(
            {
                "vehicle_type": vehicle_type,
                "outer_test_pilot": test_pilot,
                "selected_input_vars": ",".join(best_input_vars),
                "selected_window_size": best_window_size,
                "selected_num_epochs": selected_num_epochs,
                "num_test_runs": int(len(test_runs)),
                "num_test_windows": int(len(pred_df)),
                "test_accuracy": float(fold_accuracy),
            }
        )

        inner_search_df = inner_search_df.copy()
        inner_search_df["outer_test_pilot"] = test_pilot
        inner_search_tables.append(inner_search_df)

        if not inner_fold_df.empty:
            inner_fold_df = inner_fold_df.copy()
            inner_fold_df["outer_test_pilot"] = test_pilot
            inner_fold_tables.append(inner_fold_df)

    if not prediction_tables:
        raise ValueError(f"No valid outer folds were completed for vehicle type {vehicle_type}.")

    combined_result = combine_prediction_tables(prediction_tables)
    prediction_df = combined_result.pop("combined_prediction_df")
    stats_result = run_basic_statistics(prediction_df)

    return {
        "vehicle_type": vehicle_type,
        "search_df": pd.concat(inner_search_tables, ignore_index=True) if inner_search_tables else pd.DataFrame(),
        "inner_fold_df": pd.concat(inner_fold_tables, ignore_index=True) if inner_fold_tables else pd.DataFrame(),
        "outer_fold_df": pd.DataFrame(outer_fold_rows),
        "test_result": combined_result,
        "prediction_df": prediction_df,
        "statistics": stats_result,
        "num_outer_folds": int(len(outer_fold_rows)),
    }


def save_vehicle_results(result: Dict, save_dir: str) -> None:
    vehicle_dir = Path(save_dir) / result["vehicle_type"]
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    if not result["search_df"].empty:
        result["search_df"].to_csv(vehicle_dir / "inner_cv_search.csv", index=False)
    if not result["inner_fold_df"].empty:
        result["inner_fold_df"].to_csv(vehicle_dir / "inner_cv_fold_scores.csv", index=False)
    if not result["outer_fold_df"].empty:
        result["outer_fold_df"].to_csv(vehicle_dir / "outer_cv_fold_scores.csv", index=False)

    result["prediction_df"].to_csv(vehicle_dir / "window_predictions.csv", index=False)
    if "aggregated_run_predictions" in result["statistics"]:
        result["statistics"]["aggregated_run_predictions"].to_csv(vehicle_dir / "run_level_confidence.csv", index=False)

    cm = result["test_result"]["confusion_matrix"]
    pd.DataFrame(cm, index=["true_off", "true_on"], columns=["pred_off", "pred_on"]).to_csv(
        vehicle_dir / "confusion_matrix.csv"
    )

    with open(vehicle_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Vehicle type: {result['vehicle_type']}\n")
        f.write(f"Outer folds (held-out pilots): {result['num_outer_folds']}\n")
        f.write(f"Overall cross-validated accuracy: {result['test_result']['accuracy']:.4f}\n")
        f.write(f"Confusion matrix:\n{result['test_result']['confusion_matrix']}\n\n")
        if not result["outer_fold_df"].empty:
            f.write("Outer-fold selections and accuracies:\n")
            f.write(result["outer_fold_df"].to_string(index=False))
            f.write("\n\n")
        f.write("Statistics:\n")
        for key, value in result["statistics"].items():
            if key == "aggregated_run_predictions":
                f.write("aggregated_run_predictions: saved to run_level_confidence.csv\n")
            else:
                f.write(f"{key}: {value}\n")


def print_summary(result: Dict) -> None:
    print("=" * 80)
    print(f"Vehicle type                 : {result['vehicle_type']}")
    print(f"Outer CV folds (pilots)      : {result['num_outer_folds']}")
    print(f"Overall CV accuracy          : {result['test_result']['accuracy']:.4f}")
    print("Overall confusion matrix:")
    print(result["test_result"]["confusion_matrix"])
    if not result["outer_fold_df"].empty:
        print("Outer-fold selections:")
        print(result["outer_fold_df"].to_string(index=False))
    print("Statistics:")
    for key, value in result["statistics"].items():
        if key == "aggregated_run_predictions":
            print("  aggregated_run_predictions: saved to CSV")
        else:
            print(f"  {key}: {value}")


def main(config: TrainConfig):
    set_seed(config.random_state)
    os.makedirs(config.save_dir, exist_ok=True)

    runs = load_runs_from_npz_dir(config.data_dir, config=config, min_run_length=config.min_run_length)
    runs_by_vehicle = defaultdict(list)
    for run in runs:
        runs_by_vehicle[run["vehicle_type"]].append(run)

    all_results = {}
    for vehicle_type, vehicle_runs in runs_by_vehicle.items():
        labels = [r["label"] for r in vehicle_runs]
        pilot_ids = get_unique_pilot_ids(vehicle_runs)

        if len(set(labels)) < 2:
            print(f"Skipping {vehicle_type}: only one class present.")
            continue
        if len(pilot_ids) < 3:
            print(f"Skipping {vehicle_type}: need at least three pilots for nested pilot-wise cross-validation.")
            continue

        result = run_vehicle_experiment(vehicle_runs, vehicle_type, config)
        all_results[vehicle_type] = result
        save_vehicle_results(result, config.save_dir)
        print_summary(result)

    if not all_results:
        print("No vehicle experiments completed. Check your data volume, labels, and pilot coverage.")
    return all_results


if __name__ == "__main__":
    CONFIG = TrainConfig(
        data_dir="data_npz",   # change this to your folder with AE2224-I .npz files
        window_sizes=(32, 64, 96, 128),
        stride_fraction=0.5,
        input_combinations=(
            ("e", "u"),
            ("e", "u", "de"),
            ("e", "u", "du"),
            ("e", "u", "de", "du"),
        ),
        batch_size=128,
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
        save_dir="results_lstm",
    )
    main(CONFIG)
