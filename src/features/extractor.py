# src/features/extractor.py
import librosa
import numpy as np

FEATURE_NAMES: list[str] = (
    [f"mfcc_{i}_mean" for i in range(1, 14)]
    + [f"mfcc_{i}_std" for i in range(1, 14)]
    + [
        "rms_mean",
        "zcr_mean",
        "centroid_mean",
        "bandwidth_mean",
        "rolloff_mean",
    ]
)


def extract_features(samples: np.ndarray, sr: int = 51200, n_mfcc: int = 13) -> list[float]:
    y = np.asarray(samples, dtype=np.float32).ravel()
    if y.size == 0 or not np.any(np.isfinite(y)) or np.allclose(y, 0.0):
        return [0.0] * len(FEATURE_NAMES)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    vec = np.concatenate([
        mfcc_mean,
        mfcc_std,
        np.array([rms, zcr, centroid, bandwidth, rolloff], dtype=np.float32),
    ])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(x) for x in vec]