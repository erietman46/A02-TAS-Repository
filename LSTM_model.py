"""
LSTM-based time-series classifier for motion-on vs motion-off behavior.

Pipeline implemented:
1. Load run-level data from AE2224-I .npz files.
2. Normalize each run separately.
3. Build labeled short windows from each run.
4. Train an LSTM classifier per vehicle type.
5. Tune window size and input-variable combination.
6. Evaluate with accuracy and confusion matrix.
7. Compare confidence scores statistically at run level.

Expected filename convention:
- ae2224I_measurement_data_subj6_C3.npz
- Subjects: subj1 ... subj6
- Conditions: C1 ... C6

Expected content per .npz file:
- e: required, tracking/error signal
- u: required, control input signal
- label: required unless provided through a manual condition-to-label mapping
- vehicle_type: optional, defaults to "all_vehicles" if absent
- time: optional
- fs: optional

Accepted aliases in .npz files:
- e, error
- u, control
- label, motion_label, motion_on, class_label, y
- vehicle_type, vehicle, simulator, aircraft
- time, t
- fs, sample_rate, sampling_rate
"""

from __future__ import annotations

import copy
import importlib.util
import os
from collections import defaultdict
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    data_dir: str = "data_py"
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
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    min_run_length: int = 128
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results_lstm"


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


AE2224I_FILENAME_RE = re.compile(
    r"^ae2224I_measurement_data_subj(?P<subject>[1-6])_C(?P<condition>[1-6])\.npz$",
    re.IGNORECASE,
)


def parse_ae2224i_filename(file_path: Path) -> Tuple[str, str]:
    match = AE2224I_FILENAME_RE.match(file_path.name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected pattern ae2224I_measurement_data_subjX_CY.npz: {file_path.name}"
        )
    subject_id = f"subj{match.group('subject')}"
    condition_id = f"C{match.group('condition')}"
    return subject_id, condition_id


def _extract_first_available(npz_data: np.lib.npyio.NpzFile, candidates: Sequence[str], required: bool = True):
    available = {k.lower(): k for k in npz_data.files}
    for key in candidates:
        source_key = available.get(key.lower())
        if source_key is not None:
            return npz_data[source_key]
    if required:
        raise KeyError(f"Missing required key. Tried: {candidates}. Available keys: {npz_data.files}")
    return None


def _scalar_from_array(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    scalar = arr.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode('utf-8')
    return scalar.item() if hasattr(scalar, 'item') else scalar


def load_single_npz_run(npz_file: Path, min_run_length: int = 1) -> Dict | None:
    subject_id, condition_id = parse_ae2224i_filename(npz_file)

    with np.load(npz_file, allow_pickle=True) as data:
        e = np.asarray(_extract_first_available(data, ('e', 'error')), dtype=np.float32).squeeze()
        u = np.asarray(_extract_first_available(data, ('u', 'control')), dtype=np.float32).squeeze()
        label_raw = _extract_first_available(data, ('label', 'motion_label', 'motion_on', 'class_label', 'y'))
        vehicle_raw = _extract_first_available(data, ('vehicle_type', 'vehicle', 'simulator', 'aircraft'), required=False)
        time_raw = _extract_first_available(data, ('time', 't'), required=False)
        fs_raw = _extract_first_available(data, ('fs', 'sample_rate', 'sampling_rate'), required=False)

    if e.ndim != 1 or u.ndim != 1:
        raise ValueError(f"Signals e and u must be 1D arrays in {npz_file.name}.")

    n = min(len(e), len(u))
    if n < min_run_length:
        return None

    label_scalar = int(_scalar_from_array(label_raw))
    vehicle_type = str(_scalar_from_array(vehicle_raw, default='all_vehicles'))

    cleaned = {
        'e': e[:n],
        'u': u[:n],
        'label': label_scalar,
        'vehicle_type': vehicle_type,
        'pilot_id': subject_id,
        'repetition_id': condition_id,
        'source_file': npz_file.name,
        'run_id': npz_file.stem,
    }

    if time_raw is not None:
        time_arr = np.asarray(time_raw, dtype=np.float32).squeeze()
        if time_arr.ndim == 1 and len(time_arr) >= n:
            cleaned['time'] = time_arr[:n]
    if fs_raw is not None:
        cleaned['fs'] = float(_scalar_from_array(fs_raw))

    return normalize_run_signals(cleaned)


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


def load_runs_from_npz_dir(data_dir: str, min_run_length: int = 1) -> List[Dict]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    npz_files = [
        p for p in sorted(data_path.glob('*.npz'))
        if AE2224I_FILENAME_RE.match(p.name)
    ]
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir} matching ae2224I_measurement_data_subjX_CY.npz"
        )

    all_runs = []
    for npz_file in npz_files:
        run = load_single_npz_run(npz_file, min_run_length=min_run_length)
        if run is not None:
            all_runs.append(run)

    expected_names = {
        f"ae2224I_measurement_data_subj{subj}_C{cond}.npz".lower()
        for subj in range(1, 7)
        for cond in range(1, 7)
    }
    found_names = {p.name.lower() for p in npz_files}
    missing_names = sorted(expected_names - found_names)
    if missing_names:
        print('Warning: the following expected files were not found:')
        for name in missing_names:
            print(f'  - {name}')

    if not all_runs:
        raise ValueError('No valid runs were loaded from the matching .npz files.')
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


def split_runs_by_id(runs: Sequence[Dict], test_size: float, val_size: float, random_state: int):
    run_ids = [r["run_id"] for r in runs]
    labels = [r["label"] for r in runs]

    train_ids, test_ids = train_test_split(
        run_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_runs = [r for r in runs if r["run_id"] in set(train_ids)]
    test_runs = [r for r in runs if r["run_id"] in set(test_ids)]

    relative_val_size = val_size / (1.0 - test_size)
    train_labels = [r["label"] for r in train_runs]
    train_ids_final, val_ids = train_test_split(
        [r["run_id"] for r in train_runs],
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_labels,
    )

    train_final = [r for r in train_runs if r["run_id"] in set(train_ids_final)]
    val_final = [r for r in train_runs if r["run_id"] in set(val_ids)]
    return train_final, val_final, test_runs


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
        "confusion_matrix": confusion_matrix(y_true, y_pred),
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
    train_runs, val_runs, test_runs = split_runs_by_id(
        runs_for_vehicle,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )

    search_results = []
    for input_vars in config.input_combinations:
        for window_size in config.window_sizes:
            train_windows = build_window_table(train_runs, window_size, input_vars, config.stride_fraction)
            val_windows = build_window_table(val_runs, window_size, input_vars, config.stride_fraction)
            model, history_df, best_epoch, best_val_acc = fit_lstm(
                train_windows=train_windows,
                val_windows=val_windows,
                config=config,
                input_size=len(input_vars),
            )
            search_results.append(
                {
                    "vehicle_type": vehicle_type,
                    "input_vars": ",".join(input_vars),
                    "window_size": window_size,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "history_df": history_df,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                }
            )

    search_df = pd.DataFrame(
        [{k: v for k, v in item.items() if k not in {"history_df", "model_state_dict"}} for item in search_results]
    ).sort_values("best_val_accuracy", ascending=False)

    best = search_results[int(search_df.index[0])]
    best_input_vars = tuple(best["input_vars"].split(","))
    best_window_size = int(best["window_size"])

    final_model = LSTMClassifier(
        input_size=len(best_input_vars),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)
    final_model.load_state_dict(best["model_state_dict"])

    test_windows = build_window_table(test_runs, best_window_size, best_input_vars, config.stride_fraction)
    test_loader = make_loader(test_windows, config.batch_size, False, config.num_workers)
    test_result = evaluate_model(final_model, test_loader, config.device)
    pred_df = build_prediction_table(test_result)
    stats_result = run_basic_statistics(pred_df)

    return {
        "vehicle_type": vehicle_type,
        "search_df": search_df.reset_index(drop=True),
        "best_input_vars": best_input_vars,
        "best_window_size": best_window_size,
        "test_result": test_result,
        "prediction_df": pred_df,
        "statistics": stats_result,
    }


def save_vehicle_results(result: Dict, save_dir: str) -> None:
    vehicle_dir = Path(save_dir) / result["vehicle_type"]
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    result["search_df"].to_csv(vehicle_dir / "hyperparameter_search.csv", index=False)
    result["prediction_df"].to_csv(vehicle_dir / "window_predictions.csv", index=False)
    if "aggregated_run_predictions" in result["statistics"]:
        result["statistics"]["aggregated_run_predictions"].to_csv(vehicle_dir / "run_level_confidence.csv", index=False)

    cm = result["test_result"]["confusion_matrix"]
    pd.DataFrame(cm, index=["true_off", "true_on"], columns=["pred_off", "pred_on"]).to_csv(vehicle_dir / "confusion_matrix.csv")

    with open(vehicle_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Vehicle type: {result['vehicle_type']}\n")
        f.write(f"Best input vars: {result['best_input_vars']}\n")
        f.write(f"Best window size: {result['best_window_size']}\n")
        f.write(f"Test accuracy: {result['test_result']['accuracy']:.4f}\n")
        f.write(f"Confusion matrix:\n{result['test_result']['confusion_matrix']}\n\n")
        f.write("Statistics:\n")
        for key, value in result["statistics"].items():
            if key == "aggregated_run_predictions":
                f.write("aggregated_run_predictions: saved to run_level_confidence.csv\n")
            else:
                f.write(f"{key}: {value}\n")


def print_summary(result: Dict) -> None:
    print("=" * 80)
    print(f"Vehicle type       : {result['vehicle_type']}")
    print(f"Best input vars    : {result['best_input_vars']}")
    print(f"Best window size   : {result['best_window_size']}")
    print(f"Test accuracy      : {result['test_result']['accuracy']:.4f}")
    print("Confusion matrix:")
    print(result["test_result"]["confusion_matrix"])
    print("Statistics:")
    for key, value in result["statistics"].items():
        if key == "aggregated_run_predictions":
            print("  aggregated_run_predictions: saved to CSV")
        else:
            print(f"  {key}: {value}")


def main(config: TrainConfig):
    set_seed(config.random_state)
    os.makedirs(config.save_dir, exist_ok=True)

    runs = load_runs_from_npz_dir(config.data_dir, min_run_length=config.min_run_length)
    runs_by_vehicle = defaultdict(list)
    for run in runs:
        runs_by_vehicle[run["vehicle_type"]].append(run)

    all_results = {}
    for vehicle_type, vehicle_runs in runs_by_vehicle.items():
        labels = [r["label"] for r in vehicle_runs]
        if len(set(labels)) < 2:
            print(f"Skipping {vehicle_type}: only one class present.")
            continue
        if len(vehicle_runs) < 6:
            print(f"Skipping {vehicle_type}: not enough runs for train/val/test split.")
            continue

        result = run_vehicle_experiment(vehicle_runs, vehicle_type, config)
        all_results[vehicle_type] = result
        save_vehicle_results(result, config.save_dir)
        print_summary(result)

    if not all_results:
        print("No vehicle experiments completed. Check your data volume and labels.")
    return all_results


if __name__ == "__main__":
    CONFIG = TrainConfig(
        data_dir="data_npz",  # folder with ae2224I_measurement_data_subjX_CY.npz files
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
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        min_run_length=128,
        num_workers=0,
        save_dir="results_lstm",
    )

    main(CONFIG)
"""
LSTM-based time-series classifier for motion-on vs motion-off behavior.

Pipeline implemented:
1. Load run-level data from AE2224-I .npz files.
2. Normalize each run separately.
3. Build labeled short windows from each run.
4. Train an LSTM classifier per vehicle type.
5. Tune window size and input-variable combination.
6. Evaluate with accuracy and confusion matrix.
7. Compare confidence scores statistically at run level.

Expected filename convention:
- ae2224I_measurement_data_subj6_C3.npz
- Subjects: subj1 ... subj6
- Conditions: C1 ... C6

Expected content per .npz file:
- e: required, tracking/error signal
- u: required, control input signal
- label: required unless provided through a manual condition-to-label mapping
- vehicle_type: optional, defaults to "all_vehicles" if absent
- time: optional
- fs: optional

Accepted aliases in .npz files:
- e, error
- u, control
- label, motion_label, motion_on, class_label, y
- vehicle_type, vehicle, simulator, aircraft
- time, t
- fs, sample_rate, sampling_rate
"""

from __future__ import annotations

import copy
import importlib.util
import os
from collections import defaultdict
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    data_dir: str = "data_py"
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
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    min_run_length: int = 128
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results_lstm"


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


AE2224I_FILENAME_RE = re.compile(
    r"^ae2224I_measurement_data_subj(?P<subject>[1-6])_C(?P<condition>[1-6])\.npz$",
    re.IGNORECASE,
)


def parse_ae2224i_filename(file_path: Path) -> Tuple[str, str]:
    match = AE2224I_FILENAME_RE.match(file_path.name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected pattern ae2224I_measurement_data_subjX_CY.npz: {file_path.name}"
        )
    subject_id = f"subj{match.group('subject')}"
    condition_id = f"C{match.group('condition')}"
    return subject_id, condition_id


def _extract_first_available(npz_data: np.lib.npyio.NpzFile, candidates: Sequence[str], required: bool = True):
    available = {k.lower(): k for k in npz_data.files}
    for key in candidates:
        source_key = available.get(key.lower())
        if source_key is not None:
            return npz_data[source_key]
    if required:
        raise KeyError(f"Missing required key. Tried: {candidates}. Available keys: {npz_data.files}")
    return None


def _scalar_from_array(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    scalar = arr.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode('utf-8')
    return scalar.item() if hasattr(scalar, 'item') else scalar


def load_single_npz_run(npz_file: Path, min_run_length: int = 1) -> Dict | None:
    subject_id, condition_id = parse_ae2224i_filename(npz_file)

    with np.load(npz_file, allow_pickle=True) as data:
        e = np.asarray(_extract_first_available(data, ('e', 'error')), dtype=np.float32).squeeze()
        u = np.asarray(_extract_first_available(data, ('u', 'control')), dtype=np.float32).squeeze()
        label_raw = _extract_first_available(data, ('label', 'motion_label', 'motion_on', 'class_label', 'y'))
        vehicle_raw = _extract_first_available(data, ('vehicle_type', 'vehicle', 'simulator', 'aircraft'), required=False)
        time_raw = _extract_first_available(data, ('time', 't'), required=False)
        fs_raw = _extract_first_available(data, ('fs', 'sample_rate', 'sampling_rate'), required=False)

    if e.ndim != 1 or u.ndim != 1:
        raise ValueError(f"Signals e and u must be 1D arrays in {npz_file.name}.")

    n = min(len(e), len(u))
    if n < min_run_length:
        return None

    label_scalar = int(_scalar_from_array(label_raw))
    vehicle_type = str(_scalar_from_array(vehicle_raw, default='all_vehicles'))

    cleaned = {
        'e': e[:n],
        'u': u[:n],
        'label': label_scalar,
        'vehicle_type': vehicle_type,
        'pilot_id': subject_id,
        'repetition_id': condition_id,
        'source_file': npz_file.name,
        'run_id': npz_file.stem,
    }

    if time_raw is not None:
        time_arr = np.asarray(time_raw, dtype=np.float32).squeeze()
        if time_arr.ndim == 1 and len(time_arr) >= n:
            cleaned['time'] = time_arr[:n]
    if fs_raw is not None:
        cleaned['fs'] = float(_scalar_from_array(fs_raw))

    return normalize_run_signals(cleaned)


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


def load_runs_from_npz_dir(data_dir: str, min_run_length: int = 1) -> List[Dict]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    npz_files = [
        p for p in sorted(data_path.glob('*.npz'))
        if AE2224I_FILENAME_RE.match(p.name)
    ]
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir} matching ae2224I_measurement_data_subjX_CY.npz"
        )

    all_runs = []
    for npz_file in npz_files:
        run = load_single_npz_run(npz_file, min_run_length=min_run_length)
        if run is not None:
            all_runs.append(run)

    expected_names = {
        f"ae2224I_measurement_data_subj{subj}_C{cond}.npz".lower()
        for subj in range(1, 7)
        for cond in range(1, 7)
    }
    found_names = {p.name.lower() for p in npz_files}
    missing_names = sorted(expected_names - found_names)
    if missing_names:
        print('Warning: the following expected files were not found:')
        for name in missing_names:
            print(f'  - {name}')

    if not all_runs:
        raise ValueError('No valid runs were loaded from the matching .npz files.')
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


def split_runs_by_id(runs: Sequence[Dict], test_size: float, val_size: float, random_state: int):
    run_ids = [r["run_id"] for r in runs]
    labels = [r["label"] for r in runs]

    train_ids, test_ids = train_test_split(
        run_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_runs = [r for r in runs if r["run_id"] in set(train_ids)]
    test_runs = [r for r in runs if r["run_id"] in set(test_ids)]

    relative_val_size = val_size / (1.0 - test_size)
    train_labels = [r["label"] for r in train_runs]
    train_ids_final, val_ids = train_test_split(
        [r["run_id"] for r in train_runs],
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_labels,
    )

    train_final = [r for r in train_runs if r["run_id"] in set(train_ids_final)]
    val_final = [r for r in train_runs if r["run_id"] in set(val_ids)]
    return train_final, val_final, test_runs


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
        "confusion_matrix": confusion_matrix(y_true, y_pred),
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
    train_runs, val_runs, test_runs = split_runs_by_id(
        runs_for_vehicle,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )

    search_results = []
    for input_vars in config.input_combinations:
        for window_size in config.window_sizes:
            train_windows = build_window_table(train_runs, window_size, input_vars, config.stride_fraction)
            val_windows = build_window_table(val_runs, window_size, input_vars, config.stride_fraction)
            model, history_df, best_epoch, best_val_acc = fit_lstm(
                train_windows=train_windows,
                val_windows=val_windows,
                config=config,
                input_size=len(input_vars),
            )
            search_results.append(
                {
                    "vehicle_type": vehicle_type,
                    "input_vars": ",".join(input_vars),
                    "window_size": window_size,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "history_df": history_df,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                }
            )

    search_df = pd.DataFrame(
        [{k: v for k, v in item.items() if k not in {"history_df", "model_state_dict"}} for item in search_results]
    ).sort_values("best_val_accuracy", ascending=False)

    best = search_results[int(search_df.index[0])]
    best_input_vars = tuple(best["input_vars"].split(","))
    best_window_size = int(best["window_size"])

    final_model = LSTMClassifier(
        input_size=len(best_input_vars),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)
    final_model.load_state_dict(best["model_state_dict"])

    test_windows = build_window_table(test_runs, best_window_size, best_input_vars, config.stride_fraction)
    test_loader = make_loader(test_windows, config.batch_size, False, config.num_workers)
    test_result = evaluate_model(final_model, test_loader, config.device)
    pred_df = build_prediction_table(test_result)
    stats_result = run_basic_statistics(pred_df)

    return {
        "vehicle_type": vehicle_type,
        "search_df": search_df.reset_index(drop=True),
        "best_input_vars": best_input_vars,
        "best_window_size": best_window_size,
        "test_result": test_result,
        "prediction_df": pred_df,
        "statistics": stats_result,
    }


def save_vehicle_results(result: Dict, save_dir: str) -> None:
    vehicle_dir = Path(save_dir) / result["vehicle_type"]
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    result["search_df"].to_csv(vehicle_dir / "hyperparameter_search.csv", index=False)
    result["prediction_df"].to_csv(vehicle_dir / "window_predictions.csv", index=False)
    if "aggregated_run_predictions" in result["statistics"]:
        result["statistics"]["aggregated_run_predictions"].to_csv(vehicle_dir / "run_level_confidence.csv", index=False)

    cm = result["test_result"]["confusion_matrix"]
    pd.DataFrame(cm, index=["true_off", "true_on"], columns=["pred_off", "pred_on"]).to_csv(vehicle_dir / "confusion_matrix.csv")

    with open(vehicle_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Vehicle type: {result['vehicle_type']}\n")
        f.write(f"Best input vars: {result['best_input_vars']}\n")
        f.write(f"Best window size: {result['best_window_size']}\n")
        f.write(f"Test accuracy: {result['test_result']['accuracy']:.4f}\n")
        f.write(f"Confusion matrix:\n{result['test_result']['confusion_matrix']}\n\n")
        f.write("Statistics:\n")
        for key, value in result["statistics"].items():
            if key == "aggregated_run_predictions":
                f.write("aggregated_run_predictions: saved to run_level_confidence.csv\n")
            else:
                f.write(f"{key}: {value}\n")


def print_summary(result: Dict) -> None:
    print("=" * 80)
    print(f"Vehicle type       : {result['vehicle_type']}")
    print(f"Best input vars    : {result['best_input_vars']}")
    print(f"Best window size   : {result['best_window_size']}")
    print(f"Test accuracy      : {result['test_result']['accuracy']:.4f}")
    print("Confusion matrix:")
    print(result["test_result"]["confusion_matrix"])
    print("Statistics:")
    for key, value in result["statistics"].items():
        if key == "aggregated_run_predictions":
            print("  aggregated_run_predictions: saved to CSV")
        else:
            print(f"  {key}: {value}")


def main(config: TrainConfig):
    set_seed(config.random_state)
    os.makedirs(config.save_dir, exist_ok=True)

    runs = load_runs_from_npz_dir(config.data_dir, min_run_length=config.min_run_length)
    runs_by_vehicle = defaultdict(list)
    for run in runs:
        runs_by_vehicle[run["vehicle_type"]].append(run)

    all_results = {}
    for vehicle_type, vehicle_runs in runs_by_vehicle.items():
        labels = [r["label"] for r in vehicle_runs]
        if len(set(labels)) < 2:
            print(f"Skipping {vehicle_type}: only one class present.")
            continue
        if len(vehicle_runs) < 6:
            print(f"Skipping {vehicle_type}: not enough runs for train/val/test split.")
            continue

        result = run_vehicle_experiment(vehicle_runs, vehicle_type, config)
        all_results[vehicle_type] = result
        save_vehicle_results(result, config.save_dir)
        print_summary(result)

    if not all_results:
        print("No vehicle experiments completed. Check your data volume and labels.")
    return all_results


if __name__ == "__main__":
    CONFIG = TrainConfig(
        data_dir="data_npz",  # folder with ae2224I_measurement_data_subjX_CY.npz files
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
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        min_run_length=128,
        num_workers=0,
        save_dir="results_lstm",
    )

    main(CONFIG)
