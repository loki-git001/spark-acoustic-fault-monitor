import argparse
import base64
import json
import socket
import time
from pathlib import Path
import numpy as np

from src.audio.segmentation import infer_label, load_kaist_signal, segment_recording


def stream_files(
    files: list[Path],
    host: str,
    port: int,
    chunk_sec: float,
    pace_sec: float,
    noise_std: float,
    loop_forever: bool,
):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print(f"[STREAMER] Listening on {host}:{port} with {len(files)} file(s)...")

        while True:
            conn, addr = srv.accept()
            print(f"[STREAMER] Connected by {addr}")
            chunk_seq = 0
            try:
                with conn:
                    while True:
                        for path in files:
                            sig, sr = load_kaist_signal(path)
                            windows = segment_recording(
                                sig, sr=sr, window_sec=chunk_sec
                            )
                            label = infer_label(path.name)

                            for win_idx, win in enumerate(windows):
                                arr = win.astype(np.float32)
                                if noise_std > 0.0:
                                    arr = arr + np.random.normal(
                                        0.0, noise_std, size=arr.shape
                                    ).astype(np.float32)

                                payload = {
                                    "ts": time.time(),
                                    "machine_id": "motor_01",
                                    "recording_id": path.stem,
                                    "true_label": label,
                                    "window_idx": int(win_idx),
                                    "chunk_seq": int(chunk_seq),
                                    "sr": int(sr),
                                    "pcm_b64": base64.b64encode(arr.tobytes()).decode(
                                        "ascii"
                                    ),
                                }
                                line = (json.dumps(payload) + "\n").encode("utf-8")
                                conn.sendall(line)
                                print(
                                    f"[STREAMER] seq={chunk_seq} rec={path.stem} "
                                    f"win={win_idx} label={label} samples={len(arr)}"
                                )
                                chunk_seq += 1

                                if pace_sec > 0.0:
                                    time.sleep(pace_sec)

                        if not loop_forever:
                            break
                    print("[STREAMER] Finished replay pass.")
            except (BrokenPipeError, ConnectionResetError):
                print("[STREAMER] Client disconnected; waiting for new connection...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay KAIST .mat audio chunks over TCP."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=None,
        help="Optional specific .mat files to replay",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--chunk-sec", type=float, default=2.0)
    parser.add_argument(
        "--pace-sec",
        type=float,
        default=1.0,
        help="Seconds between chunks (use 2.0 for real-time)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std to add per chunk",
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop files continuously until stopped"
    )
    args = parser.parse_args()

    target_files = (
        sorted(args.files) if args.files else sorted(args.input_dir.rglob("*.mat"))
    )
    if not target_files:
        raise FileNotFoundError("No .mat files found to stream.")

    stream_files(
        files=target_files,
        host=args.host,
        port=args.port,
        chunk_sec=args.chunk_sec,
        pace_sec=args.pace_sec,
        noise_std=args.noise_std,
        loop_forever=args.loop,
    )
