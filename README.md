# Acoustic Fault Diagnosis & Maintenance Assistant (Streaming + MLlib Prototype)

A real-time rotating-machine acoustic monitoring prototype built with **Apache Spark Structured Streaming**, **Spark MLlib**, **HDFS**, and **Librosa**. 

This implementation focuses on the core ingestion, acoustic feature engineering, distributed ML classification, and stateful fault-event triggering pipeline. Audio chunks from the **KAIST Rotating Machine Acoustic Dataset** are streamed over a live TCP socket into Spark, transformed into spectral and time-domain features, classified using a trained Spark MLlib `PipelineModel`, and aggregated via windowed threshold + cooldown rules to emit actionable fault alerts.

---

## Architecture Overview

```text
[KAIST Acoustic .mat Files]
       │
       ├──► [Offline / Batch Pipeline (HDFS + Spark MLlib)]
       │      ├── Load Signal.y_values.values (51.2 kHz) & segment into 2-sec windows
       │      ├── Leak-free recording-level train/test split
       │      ├── Extract 31 acoustic features (MFCCs, RMS, ZCR, Spectral stats)
       │      ├── Store feature Parquet tables on HDFS
       │      └── Train & save Spark MLlib PipelineModel to HDFS
       │
       └──► [Real-Time Streaming Pipeline (TCP Socket -> Spark Structured Streaming)]
              ├── tcp_audio_streamer.py streams 2-sec base64 PCM chunks over 127.0.0.1:9999
              ├── Spark `readStream.format("socket")` ingests NDJSON chunks
              ├── UDF extracts acoustic features -> StandardScaler + RandomForest inference
              └── FaultEventDetector triggers alerts (>= N of last M windows + cooldown)
```

---

## Project Structure

```text
acoustic-fault-stream/
│
├── data/
│   ├── raw/                        # KAIST acoustic .mat files (e.g., 0Nm_Normal.mat, 0Nm_BPFI_03.mat)
│   └── processed/                  # Local intermediate artifacts (if needed)
│
├── models/                         # Optional local model backups
│
├── notebooks/
│   └── model_experiments.ipynb     # Spark MLlib model comparison & CV tuning (RF, LR, LinearSVC, MLP)
│
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── segmentation.py         # MATLAB struct parser, label mapping, leak-free windowing
│   ├── features/
│   │   ├── __init__.py
│   │   └── extractor.py            # 13 MFCCs (mean/std) + RMS, ZCR, Centroid, Bandwidth, Rolloff
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── train_mllib.py          # Baseline Spark MLlib Random Forest training & evaluation
│   │   └── save_best_model.py      # CLI script to train, evaluate, and export best PipelineModel to HDFS
│   └── streaming/
│       ├── __init__.py
│       ├── tcp_echo_test.py        # Socket-to-Spark connectivity smoke test
│       ├── tcp_audio_streamer.py   # Replays 2-second .mat audio chunks over TCP socket
│       └── spark_fault_stream.py   # Spark Structured Streaming inference + FaultEventDetector
│
├── pyproject.toml                  # Project dependencies managed by uv
├── uv.lock                         # Locked dependency versions
└── README.md
```

---

## Prerequisites & Environment Setup (`uv`)

This project runs on **WSL2 / Linux** with **Apache Spark**, **Hadoop HDFS** (`hdfs://localhost:9000`), and **`uv`** for Python dependency management.

1. **Sync the Python environment using `uv`:**
   ```bash
   cd ~/projects/acoustic-fault-stream
   uv sync
   ```
   *(You can either run commands prefixed with `uv run` or activate the virtual environment via `source .venv/bin/activate`.)*

2. **Verify Java, Spark, and HDFS services:**
   ```bash
   which java spark-submit hdfs
   hdfs dfs -mkdir -p /spark/data/kaist /spark/models /spark/checkpoints
   ```

3. **Dataset Placement:**
   Place your KAIST Rotating Machine Acoustic `.mat` files inside `data/raw/`. Each `.mat` file contains a `Signal` MATLAB struct sampled at **51.2 kHz** (`3,072,000` samples = 60 seconds per recording), which is segmented into non-overlapping **2-second windows** (`102,400` samples).

---

## Step-by-Step Execution Guide

### 1. Extract Features & Load Train/Test Parquet to HDFS
Run the leak-free segmentation and feature extraction job to populate HDFS feature tables (`31` features + metadata columns):

```bash
PYTHONPATH=. uv run python src/features/extractor.py  # Or run the batch feature script below
```

To build the train/test Parquet datasets directly on HDFS:
```bash
PYTHONPATH=. uv run python - <<'EOF'
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from src.audio.segmentation import build_leak_free_windows
from src.features.extractor import FEATURE_NAMES, extract_features

files = sorted(Path("data/raw").rglob("*.mat"))
train_w, test_w = build_leak_free_windows(files, window_sec=2.0, test_ratio=0.5, seed=42)
col_names = [f"f_{i}" for i in range(len(FEATURE_NAMES))]

def to_rows(windows):
    return [(w.recording_id, int(w.window_idx), w.label, int(w.sr), *extract_features(w.samples, sr=w.sr)) for w in windows]

schema = StructType(
    [
        StructField("recording_id", StringType(), False),
        StructField("window_idx", IntegerType(), False),
        StructField("label", StringType(), False),
        StructField("sr", IntegerType(), False),
    ] + [StructField(c, DoubleType(), False) for c in col_names]
)

spark = SparkSession.builder.appName("KAISTFeatureLoad").config("spark.sql.shuffle.partitions", "4").getOrCreate()
spark.createDataFrame(to_rows(train_w), schema=schema).write.mode("overwrite").parquet("hdfs://localhost:9000/spark/data/kaist/train")
spark.createDataFrame(to_rows(test_w), schema=schema).write.mode("overwrite").parquet("hdfs://localhost:9000/spark/data/kaist/test")
spark.stop()
EOF
```

### 2. Experiment & Train the Spark MLlib PipelineModel
* **Interactive Experiments:** Open `notebooks/model_experiments.ipynb` to compare `RandomForestClassifier`, `LogisticRegression` (multinomial), `OneVsRest(LinearSVC)`, and `MultilayerPerceptronClassifier` with `StandardScaler` and cross-validation.
* **Export Best Model to HDFS:** Run `src/ml/save_best_model.py` to fit the end-to-end `Pipeline` (`StringIndexer` $\rightarrow$ `VectorAssembler` $\rightarrow$ `StandardScaler` $\rightarrow$ `Classifier`), evaluate metrics/confusion matrix, and save the `PipelineModel` to HDFS:
  ```bash
  PYTHONPATH=. uv run python src/ml/save_best_model.py \
    --model rf \
    --num-trees 100 \
    --max-depth 4 \
    --output-path hdfs://localhost:9000/spark/models/kaist_fault_best
  ```

### 3. Run Real-Time Streaming Inference & Fault-Event Detection
Open **two WSL terminals** inside `~/projects/acoustic-fault-stream`.

**Terminal 1 — Start Spark Structured Streaming Consumer:**
```bash
# Clear previous streaming checkpoint if restarting cleanly
hdfs dfs -rm -r -f /spark/checkpoints/acoustic_fault_stream

# Cap worker thread pools in WSL for stability
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/numba_cache

PYTHONPATH=. uv run python src/streaming/spark_fault_stream.py \
  --host 127.0.0.1 \
  --port 9999 \
  --model-path hdfs://localhost:9000/spark/models/kaist_fault_best \
  --n-required 3 \
  --m-window 5 \
  --prob-threshold 0.60 \
  --cooldown-sec 10.0
```

**Terminal 2 — Start TCP Audio Replay Server:**
```bash
PYTHONPATH=. uv run python src/streaming/tcp_audio_streamer.py \
  --input-dir data/raw \
  --host 127.0.0.1 \
  --port 9999 \
  --pace-sec 2.0 \
  --noise-std 0.001
```

---

## Feature Engineering Summary

Each 2-second window (`102,400` samples at `51.2 kHz`) is transformed into **31 numeric features**:
* **MFCC Means (`f_0` – `f_12`):** Mean across frames for the first 13 Mel-Frequency Cepstral Coefficients.
* **MFCC Standard Deviations (`f_13` – `f_25`):** Standard deviation across frames for the first 13 MFCCs.
* **Root Mean Square Energy (`f_26`):** Mean frame energy (`librosa.feature.rms`).
* **Zero Crossing Rate (`f_27`):** Mean waveform sign-change rate (`librosa.feature.zero_crossing_rate`).
* **Spectral Centroid (`f_28`):** Center of mass of the spectrum (`librosa.feature.spectral_centroid`).
* **Spectral Bandwidth (`f_29`):** Weighted spread around the spectral centroid (`librosa.feature.spectral_bandwidth`).
* **Spectral Rolloff (`f_30`):** Frequency below which 85% of spectral energy lies (`librosa.feature.spectral_rolloff`).

---

## Fault-Event Triggering Logic

To prevent single-window acoustic spikes from generating noisy alerts, `FaultEventDetector` tracks a sliding window per `machine_id`:
* **Persistence Rule:** Triggers only when at least `N` of the last `M` windows (default: `3` of `5`) predict the same non-normal condition with probability $\ge$ `prob_threshold` (default: `0.60`).
* **Cooldown Suppression:** Suppresses duplicate alerts for the same `(machine_id, condition)` pair for `cooldown_sec` (default: `10.0` seconds).

---

## Known Prototype Limitations & Next Steps

1. **Cross-Recording Generalization:** When splitting whole recordings across fault severities (e.g., `0Nm_BPFI_03.mat` vs. `0Nm_BPFI_10.mat`) with a limited number of `.mat` files, raw spectral/amplitude shifts between recordings can limit unseen-file test accuracy (~20% baseline on 3 classes). Incorporating more recordings across load/severity conditions and per-window spectral normalization will improve generalization.
2. **WSL Memory & Worker Tuning:** Processing `51.2 kHz` audio chunks inside PySpark Python UDFs requires bounding thread pools (`OMP_NUM_THREADS=1`), dropping raw base64 payloads immediately after feature extraction, and configuring adequate WSL2 memory in `%USERPROFILE%\.wslconfig`.
3. **Future Extensions:** Connect the `FAULT_EVENT_TRIGGERED` JSON output sink to a downstream **LangGraph + RAG** maintenance assistant to retrieve bearing/shaft documentation and generate grounded inspection reports.