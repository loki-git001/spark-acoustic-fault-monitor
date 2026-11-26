import argparse
import json
import socket
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


def run_sender(host: str, port: int, count: int = 8, interval: float = 1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print(f"[SENDER] Listening on {host}:{port} ...")
        conn, addr = srv.accept()
        print(f"[SENDER] Connected by {addr}")
        with conn:
            for i in range(count):
                msg = {
                    "ts": time.time(),
                    "machine_id": "motor_01",
                    "chunk_id": i,
                    "status": "ok",
                }
                conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
                print(f"[SENDER] Sent chunk {i}")
                time.sleep(interval)
        print("[SENDER] Done sending test chunks.")


def run_receiver(host: str, port: int, checkpoint_dir: str):
    spark = (
        SparkSession.builder
        .appName("Hour1SocketSmokeTest")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("ts", DoubleType(), True),
        StructField("machine_id", StringType(), True),
        StructField("chunk_id", DoubleType(), True),
        StructField("status", StringType(), True),
    ])

    raw = (
        spark.readStream
        .format("socket")
        .option("host", host)
        .option("port", port)
        .load()
    )

    parsed = (
        raw.select(F.from_json(F.col("value"), schema).alias("d"))
        .select("d.*")
    )

    query = (
        parsed.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("checkpointLocation", checkpoint_dir)
        .start()
    )

    query.awaitTermination(timeout=20)
    query.stop()
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sender", "receiver"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--checkpoint", default="hdfs://localhost:9000/checkpoints/hour1_socket_test")
    args = parser.parse_args()

    if args.mode == "sender":
        run_sender(args.host, args.port)
    else:
        run_receiver(args.host, args.port, args.checkpoint)