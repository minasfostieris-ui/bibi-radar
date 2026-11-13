#!/usr/bin/env python3
"""
bibi_radar.py — live screen radar that circles Netanyahu in whatever plays on
your screen. Pure satire/meme tooling; no data leaves your machine, nothing is
stored, no identity database, not surveillance.

ONE COMMAND to run (auto-installs deps on first launch, auto-downloads the
model from the public Hugging Face repo):

    python bibi_radar.py

Backends:
  --backend auto  (default) GPU if torch+HIP is visible, else OpenVINO CPU.
  --backend gpu   torch-HIP + ultralytics, fp16, threaded capture (AMD ROCm).
  --backend cpu   OpenVINO CPU (the original, works everywhere).

On AMD RDNA2 (e.g. RX 6700 XT) the gfx1030 disguise is applied automatically
(HSA_OVERRIDE_GFX_VERSION=10.3.0). For GPU, run under a ROCm torch venv, e.g.:
    ~/.venvs/rocm/bin/python bibi_radar.py --backend gpu

Works on Windows (Python 3.10-3.13, x86_64) and Linux (X11 or Wayland+grim).
Press q in the radar window to quit. Optional: --conf 0.5 --run-for 12
"""
import argparse
import importlib.util
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import warnings

# ultralytics deprecation chatter (half->quantize) is noise for a meme overlay
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")

# gfx1030 disguise for RDNA2 cards; must be set before torch/HIP initializes.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

HF_REPO = "minas2025/bibi-circleator"
OV_DIR = "openvino"          # folder inside the repo holding the CPU build
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "bibi-radar")
BASE_DEPS = {"numpy": "numpy", "cv2": "opencv-python",
             "huggingface_hub": "huggingface_hub"}
CPU_DEPS = {"openvino": "openvino", "mss": "mss"}
GPU_DEPS = {"ultralytics": "ultralytics", "mss": "mss"}


def gpu_available():
    """True if a HIP/CUDA torch device is importable and visible."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def bootstrap(backend):
    """First-run magic: pip-install anything missing, no separate step needed."""
    wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and shutil.which("grim")
    want = dict(BASE_DEPS)
    want.update(CPU_DEPS if backend == "cpu" else GPU_DEPS)
    if wayland:
        want.pop("mss", None)
    want = {m: p for m, p in want.items() if importlib.util.find_spec(m) is None}
    if not want:
        return
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    cmd = [sys.executable, "-m", "pip", "install"]
    if not in_venv:
        cmd.append("--user")
    print(f"first run: installing {', '.join(want.values())} ...", flush=True)
    subprocess.check_call(cmd + list(want.values()))


def get_model(backend):
    """Fetch the right build once; cached in ~/.cache/bibi-radar."""
    if backend == "gpu":
        pt = os.path.join(CACHE, "bibi-circle-v3.pt")
        if not os.path.exists(pt):
            from huggingface_hub import hf_hub_download
            print("downloading GPU model (5.5 MB, one time) ...", flush=True)
            hf_hub_download(HF_REPO, "bibi-circle-v3.pt", local_dir=CACHE)
        return pt
    xml = os.path.join(CACHE, OV_DIR, "bibi-circle-v3.xml")
    if not os.path.exists(xml):
        from huggingface_hub import hf_hub_download
        print("downloading model (11 MB, one time) ...", flush=True)
        for f in ("bibi-circle-v3.xml", "bibi-circle-v3.bin"):
            hf_hub_download(HF_REPO, f"{OV_DIR}/{f}", local_dir=CACHE)
    return xml


def make_grabber():
    """Pick a screen-capture backend: mss everywhere, grim on Wayland."""
    import numpy as np
    import cv2
    wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    if wayland and shutil.which("grim"):
        def grab_wayland():
            raw = subprocess.run(["grim", "-t", "ppm", "-"],
                                 capture_output=True, timeout=5).stdout
            return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return grab_wayland
    try:
        import mss
    except ImportError:
        sys.exit("no screen backend found. On Wayland install grim; otherwise "
                 "re-run so mss can install. On Windows mss is automatic.")
    sct = mss.mss()
    mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    def grab_mss():
        shot = np.array(sct.grab(mon))[:, :, :3]      # BGRA -> BGR
        return shot
    return grab_mss


class FrameSource:
    """Threaded latest-frame-wins queue so capture overlaps inference."""
    def __init__(self, grab):
        self.q = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        def loop():
            while not self.stop.is_set():
                try:
                    img = grab()
                except Exception:
                    continue
                if img is None:
                    continue
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(img)
        threading.Thread(target=loop, daemon=True).start()

    def read(self, timeout=1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class RadarCPU:
    IMGSZ = 960

    def __init__(self, conf):
        import numpy as np
        import openvino as ov
        self.np = np
        core = ov.Core()
        self.net = core.compile_model(core.read_model(get_model("cpu")), "CPU")
        self.conf = conf

    def run(self, img):
        np = self.np
        cv2 = __import__("cv2")
        conf = self.conf
        h, w = img.shape[:2]
        s = min(self.IMGSZ / h, self.IMGSZ / w)
        nh, nw = int(h * s), int(w * s)
        canvas = np.full((self.IMGSZ, self.IMGSZ, 3), 114, np.uint8)
        top = (self.IMGSZ - nh) // 2
        left = (self.IMGSZ - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = self.net(blob)[0][0]
        nc = pred.shape[0] - 4
        # 3-class argmax: bibi must clear CONF and outvote female + other_male.
        if nc >= 3:
            rival = np.maximum(pred[5], pred[6])
            sel = (pred[4] > conf) & (pred[4] >= rival)
        elif nc == 2:
            sel = (pred[4] > conf) & (pred[4] > pred[5])
        else:
            sel = pred[4] > conf
        if not sel.any():
            return []
        cand = pred[:, sel].T
        boxes = np.empty_like(cand[:, :4])
        boxes[:, 0] = (cand[:, 0] - cand[:, 2] / 2 - left) / s
        boxes[:, 1] = (cand[:, 1] - cand[:, 3] / 2 - top) / s
        boxes[:, 2] = (cand[:, 0] + cand[:, 2] / 2 - left) / s
        boxes[:, 3] = (cand[:, 1] + cand[:, 3] / 2 - top) / s
        scores = cand[:, 4]
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            ua = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) +
                  (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]) - inter)
            order = rest[np.where(ua > 0, inter / ua, 1.0) < 0.45]
        return [list(boxes[i]) + [float(scores[i])] for i in keep]


class RadarGPU:
    IMGSZ = 960

    def __init__(self, conf):
        import torch
        from ultralytics import YOLO
        self.dev = torch.device("cuda")
        self.model = YOLO(get_model("gpu"))
        self.model.to(self.dev)
        self.conf = conf
        # warmup so the first real frame isn't a 400ms cold start
        import numpy as np
        self.model(np.zeros((720, 1280, 3), np.uint8), device=self.dev,
                   imgsz=self.IMGSZ, half=True, verbose=False)

    def run(self, img):
        r = self.model(img, device=self.dev, imgsz=self.IMGSZ, half=True,
                       conf=self.conf, verbose=False)[0]
        keep = []
        if r.boxes is not None and len(r.boxes):
            m = r.boxes.cls.int() == 0        # bibi only; ultralytics argmax built in
            for b, c in zip(r.boxes.xyxy[m].tolist(), r.boxes.conf[m].tolist()):
                keep.append(b + [c])
        return keep


def main():
    ap = argparse.ArgumentParser(description="circle him on your screen")
    ap.add_argument("--conf", type=float, default=0.45, help="min bibi score (default 0.45)")
    ap.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--run-for", type=float, default=0, help="auto-exit after N seconds (0 = until q)")
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "gpu" if gpu_available() else "cpu"
    if backend == "gpu" and not gpu_available():
        print("GPU requested but no HIP/CUDA torch device; falling back to CPU.", flush=True)
        backend = "cpu"

    bootstrap(backend)
    import cv2
    r = RadarGPU(args.conf) if backend == "gpu" else RadarCPU(args.conf)
    src = FrameSource(make_grabber())

    title = f"BIG YAHU RADAR {backend.upper()} (q quits)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    print(f"radar up [{backend}] - play the video. q quits.", flush=True)
    lats = []
    fps_sm, last_report, start = 0.0, 0.0, time.time()
    while True:
        if args.run_for and time.time() - start > args.run_for:
            break
        img = src.read()
        if img is None:
            continue
        t0 = time.time()
        hits = r.run(img)
        disp = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        for (x1, y1, x2, y2, sc) in hits:
            cx, cy = int((x1 + x2) / 4), int((y1 + y2) / 4)
            rad = int(((x2 - x1) + (y2 - y1)) / 8 * 1.3)
            cv2.circle(disp, (cx, cy), rad, (0, 214, 255), 4, cv2.LINE_AA)
            cv2.putText(disp, f"BIG YAHU {int(sc * 100)}%",
                        (cx - rad, max(24, cy - int(rad * 1.1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 214, 255), 2, cv2.LINE_AA)
        dt = (time.time() - t0) * 1000
        lats.append(dt)
        if len(lats) > 20:
            lats.pop(0)
        fps_sm = fps_sm * 0.9 + (1000 / max(dt, 1e-4)) * 0.1
        now = time.time()
        if now - last_report > 2.0:
            print(f"[{backend}] ~{fps_sm:.0f} fps | loop {sum(lats)/len(lats):.0f} ms", flush=True)
            last_report = now
        cv2.putText(disp, f"{fps_sm:.0f} fps", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.imshow(title, disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    src.stop.set()
    cv2.destroyAllWindows()
    print(f"[{backend}] final ~{fps_sm:.0f} fps", flush=True)


if __name__ == "__main__":
    main()