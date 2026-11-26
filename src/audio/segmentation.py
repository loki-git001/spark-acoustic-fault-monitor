# src/audio/segmentation.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat


@dataclass
class AudioWindow:
    recording_id: str
    split: str  # "train" or "test"
    window_idx: int
    label: str
    sr: int
    samples: np.ndarray


def infer_label(filename: str) -> str:
    name = filename.lower()
    if "normal" in name:
        return "normal"
    if "bpfi" in name or "inner" in name:
        return "inner_race"
    if "bpfo" in name or "outer" in name:
        return "outer_race"
    if "misalign" in name:
        return "misalignment"
    if "unbalan" in name:
        return "unbalance"
    return "unknown"


def load_kaist_signal(mat_path: Path, default_sr: int = 51200) -> tuple[np.ndarray, int]:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    sr = default_sr

    if "Signal" in data:
        sig_obj = data["Signal"]
        x_val = getattr(sig_obj, "x_values", None)
        inc = getattr(x_val, "increment", None)
        if inc and float(inc) > 0:
            sr = int(round(1.0 / float(inc)))

        y_val = getattr(sig_obj, "y_values", None)
        vals = getattr(y_val, "values", None)
        if vals is not None:
            return np.squeeze(np.asarray(vals)).astype(np.float32), sr

    # Fallback if a file ever has a plain numeric array
    keys = [k for k in data.keys() if not k.startswith("__")]
    numeric_arrays = [
        np.squeeze(np.asarray(data[k]))
        for k in keys
        if isinstance(data[k], np.ndarray) and np.issubdtype(np.asarray(data[k]).dtype, np.number)
    ]
    if not numeric_arrays:
        raise ValueError(f"Could not extract numeric signal from {mat_path}")
    arr = max(numeric_arrays, key=lambda a: a.size)
    return arr.astype(np.float32), sr


def segment_recording(
    signal: np.ndarray,
    sr: int = 51200,
    window_sec: float = 2.0,
) -> list[np.ndarray]:
    win_len = int(round(sr * window_sec))
    if win_len <= 0 or len(signal) < win_len:
        return []
    n_wins = len(signal) // win_len
    return [signal[i * win_len : (i + 1) * win_len] for i in range(n_wins)]


def build_leak_free_windows(
    mat_files: list[Path],
    window_sec: float = 2.0,
    test_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list[AudioWindow], list[AudioWindow]]:
    """
    - For classes with >= 2 files (e.g., inner_race, outer_race): splits by whole file
      so train and test come from completely separate recordings.
    - For single-file classes (e.g., normal in your current raw dir): splits contiguous
      time windows (first portion train, last portion test) so test windows are still
      evaluated on that class rather than having 0 test samples.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Path]] = {}
    for p in sorted(mat_files):
        lbl = infer_label(p.name)
        if lbl != "unknown":
            by_label.setdefault(lbl, []).append(p)

    train_windows: list[AudioWindow] = []
    test_windows: list[AudioWindow] = []

    for lbl, group in sorted(by_label.items()):
        if len(group) >= 2:
            perm = rng.permutation(len(group))
            shuffled = [group[i] for i in perm]
            n_test = max(1, int(round(len(shuffled) * test_ratio)))
            n_test = min(n_test, len(shuffled) - 1)
            test_files = set(shuffled[:n_test])

            for path in shuffled:
                sig, sr = load_kaist_signal(path)
                chunks = segment_recording(sig, sr=sr, window_sec=window_sec)
                split_name = "test" if path in test_files else "train"
                target = test_windows if split_name == "test" else train_windows
                for idx, chunk in enumerate(chunks):
                    target.append(
                        AudioWindow(
                            recording_id=path.stem,
                            split=split_name,
                            window_idx=idx,
                            label=lbl,
                            sr=sr,
                            samples=chunk,
                        )
                    )
        else:
            # Single-file class: contiguous block split (no random interleave)
            path = group[0]
            sig, sr = load_kaist_signal(path)
            chunks = segment_recording(sig, sr=sr, window_sec=window_sec)
            split_idx = int(len(chunks) * (1.0 - test_ratio))
            split_idx = max(1, min(split_idx, len(chunks) - 1))
            for idx, chunk in enumerate(chunks):
                is_test = idx >= split_idx
                win = AudioWindow(
                    recording_id=path.stem,
                    split="test" if is_test else "train",
                    window_idx=idx,
                    label=lbl,
                    sr=sr,
                    samples=chunk,
                )
                (test_windows if is_test else train_windows).append(win)

    return train_windows, test_windows