import argparse
import base64
from collections import deque
import json
import time
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StandardScalerModel, StringIndexerModel, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.features.extractor import FEATURE_NAMES, extract_features


class FaultEventDetector:
    """Triggers a fault event when >= n_required of the last m_window predictions
    match a non-normal fault class with probability >= prob_threshold,
    respecting a per-(machine_id, condition) cooldown window."""

    def __init__(
        self,
        n_required: int = 3,
        m_window: int = 5,
        prob_threshold: float = 0.60,
        cooldown_sec: float = 10.0,
    ):
        self.n_required = n_required
        self.m_window = m_window
        self.prob_threshold = prob_threshold
        self.cooldown_sec = cooldown_sec
        self.history: dict[str, deque[tuple[str, float, float]]] = {}
        self.last_alert_ts: dict[tuple[str, str], float] = {}

    def update(
        self, machine_id: str, pred_label: str, prob: float, ts: float
    ) -> dict | None:
        buf = self.history.setdefault(machine_id, deque(maxlen=self.m_window))
        buf.append((pred_label, prob, ts))

        if pred_label in ("normal", "unknown") or prob < self.prob_threshold:
            return None

        hits = sum(
            1 for lbl, p, _ in buf if lbl == pred_label and p >= self.prob_threshold
        )
        if hits < self.n_required:
            return None

        key = (machine_id, pred_label)
        prev_ts = self.last_alert_ts.get(key, -1e18)
        if (ts - prev_ts) < self.cooldown_sec:
            return None

        self.last_alert_ts[key] = ts
        return {
            "event": "FAULT_EVENT_TRIGGERED",
            "machine_id": machine_id,
            "condition": pred_label,
            "confidence": round(prob, 4),
            "hits_in_last_m": f"{hits}/{len(buf)}",
            "threshold": self.prob_threshold,
            "cooldown_sec": self.cooldown_sec,
            "event_ts": ts,
        }


def make_feature_udf():
    @F.udf(returnType=ArrayType(DoubleType()))
    def _extract_udf(pcm_b64: str, sr: int):
        if not pcm_b64:
            return [0.0] * len(FEATURE_NAMES)
        raw = base64.b64decode(pcm_b64.encode("ascii"))
        samples = np.frombuffer(raw, dtype=np.float32)
        sample_rate = int(sr) if sr and int(sr) > 0 else 51200
        return extract_features(samples, sr=sample_rate, n_mfcc=13)

    return _extract_udf


def main():
    parser = argparse.ArgumentParser(
        description="Spark Structured Streaming acoustic fault inference."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--model-path",
        default="hdfs://localhost:9000/spark/models/kaist_fault_best",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="hdfs://localhost:9000/spark/checkpoints/acoustic_fault_stream",
    )
    parser.add_argument("--n-required", type=int, default=3)
    parser.add_argument("--m-window", type=int, default=5)
    parser.add_argument("--prob-threshold", type=float, default=0.60)
    parser.add_argument("--cooldown-sec", type=float, default=10.0)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("Hours5and6AcousticFaultStream")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.network.timeout", "120s")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    pipeline_model = PipelineModel.load(args.model_path)
    indexer_model = next(
        s for s in pipeline_model.stages if isinstance(s, StringIndexerModel)
    )
    scaler_model = next(
        s for s in pipeline_model.stages if isinstance(s, StandardScalerModel)
    )
    rf_model = next(
        s
        for s in pipeline_model.stages
        if isinstance(s, RandomForestClassificationModel)
    )

    labels = list(indexer_model.labels)
    feature_cols = [f"f_{i}" for i in range(len(FEATURE_NAMES))]

    incoming_schema = StructType(
        [
            StructField("ts", DoubleType(), True),
            StructField("machine_id", StringType(), True),
            StructField("recording_id", StringType(), True),
            StructField("true_label", StringType(), True),
            StructField("window_idx", IntegerType(), True),
            StructField("chunk_seq", IntegerType(), True),
            StructField("sr", IntegerType(), True),
            StructField("pcm_b64", StringType(), True),
        ]
    )

    socket_df = (
        spark.readStream.format("socket")
        .option("host", args.host)
        .option("port", args.port)
        .load()
    )

    feat_udf = make_feature_udf()

    parsed = (
        socket_df.select(F.from_json(F.col("value"), incoming_schema).alias("d"))
        .select("d.*")
        .filter(F.col("pcm_b64").isNotNull())
        .withColumn("feat_vec", feat_udf(F.col("pcm_b64"), F.col("sr")))
        .drop("pcm_b64")
    )

    for idx in range(len(FEATURE_NAMES)):
        parsed = parsed.withColumn(f"f_{idx}", F.col("feat_vec").getItem(idx))

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    assembled = assembler.transform(parsed)
    scaled = scaler_model.transform(assembled)
    scored = rf_model.transform(scaled)

    @F.udf(returnType=StringType())
    def _idx_to_label(pred_idx: float):
        if pred_idx is None:
            return "unknown"
        idx = int(pred_idx)
        return labels[idx] if 0 <= idx < len(labels) else "unknown"

    @F.udf(returnType=DoubleType())
    def _pred_prob(prob_vec, pred_idx: float):
        if prob_vec is None or pred_idx is None:
            return 0.0
        idx = int(pred_idx)
        return float(prob_vec[idx]) if 0 <= idx < len(prob_vec) else 0.0

    predictions_df = (
        scored.withColumn("pred_label", _idx_to_label(F.col("prediction")))
        .withColumn("pred_prob", _pred_prob(F.col("probability"), F.col("prediction")))
        .select(
            "ts",
            "chunk_seq",
            "machine_id",
            "recording_id",
            "window_idx",
            "true_label",
            "pred_label",
            "pred_prob",
        )
    )

    detector = FaultEventDetector(
        n_required=args.n_required,
        m_window=args.m_window,
        prob_threshold=args.prob_threshold,
        cooldown_sec=args.cooldown_sec,
    )

    def handle_microbatch(batch_df, batch_id: int):
        rows = batch_df.collect()
        for r in rows:
            ts_val = float(r["ts"]) if r["ts"] is not None else time.time()
            prob_val = float(r["pred_prob"]) if r["pred_prob"] is not None else 0.0
            pred_lbl = r["pred_label"] or "unknown"
            machine = r["machine_id"] or "motor_01"

            print(
                f"[batch={batch_id} seq={r['chunk_seq']} rec={r['recording_id']} win={r['window_idx']}] "
                f"true={r['true_label']} -> pred={pred_lbl} (p={prob_val:.3f})"
            )

            event = detector.update(
                machine_id=machine,
                pred_label=pred_lbl,
                prob=prob_val,
                ts=ts_val,
            )
            if event:
                print("==========================================")
                print("ALERT:", json.dumps(event))
                print("==========================================")

    query = (
        predictions_df.writeStream.foreachBatch(handle_microbatch)
        .option("checkpointLocation", args.checkpoint_path)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
