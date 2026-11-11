#!/usr/bin/env python3
"""
bibi_radar.py — live screen radar that circles Netanyahu in whatever plays on
your screen. Pure satire/meme tooling; no data leaves your machine, nothing is
stored, no identity database, not surveillance.

ONE COMMAND to run (auto-installs deps on first launch, auto-downloads the
model from the public Hugging Face repo):

    python bibi_radar.py

Works on Windows (Python 3.10-3.13, x86_64) and Linux (X11 or Wayland+grim).
Press q in the radar window to quit. Optional: --conf 0.5
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time

HF_REPO = "minas2025/bibi-circleator"
OV_DIR = "openvino"          # folder inside the repo holding the CPU build
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "bibi-radar")
DEPS = {"numpy": "numpy", "cv2": "opencv-python", "openvino": "openvino",
        "mss": "mss", "huggingface_hub": "huggingface_hub"}


def bootstrap():
    """First-run magic: pip-install anything missing, no separate step needed."""
    # grim covers Wayland; mss only matters on Windows/macOS/X11
    wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and shutil.which("grim")
    want = {m: p for m, p in DEPS.items() if importlib.util.find_spec(m) is None}
    if wayland:
        want.pop("mss", None)
    if not want:
        return
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    cmd = [sys.executable, "-m", "pip", "install"]
    if not in_venv:
        cmd.append("--user")
    print(f"first run: installing {', '.join(want.values())} ...", flush=True)
    subprocess.check_call(cmd + list(want.values()))


def get_model():
    """Fetch the OpenVINO CPU build once; cached in ~/.cache/bibi-radar."""
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


class Radar:
    IMGSZ = 960

    def __init__(self, conf):
        import numpy as np
        import openvino as ov
        self.np = np
        core = ov.Core()
        self.net = core.compile_model(core.read_model(get_model()), "CPU")

        def run(img):
            np = self.np
            cv2 = __import__("cv2")
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
        self.run = run


def main():
    ap = argparse.ArgumentParser(description="circle him on your screen")
    ap.add_argument("--conf", type=float, default=0.45, help="min bibi score (default 0.45)")
    args = ap.parse_args()

    bootstrap()
    import numpy as np
    import cv2
    r = Radar(args.conf)
    grab = make_grabber()

    title = "BIG YAHU RADAR (q quits)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    print("radar up - play the video. q quits.", flush=True)
    lats = []
    while True:
        t0 = time.time()
        img = grab()
        if img is None:
            time.sleep(0.1)
            continue
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
        cv2.putText(disp, f"{1000 / (sum(lats) / len(lats)):.0f} fps", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.imshow(title, disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
