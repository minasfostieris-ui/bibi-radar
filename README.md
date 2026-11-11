# bibi-radar 🎯

A live screen radar that draws a big orange **BIG YAHU 92%** circle around
Benjamin Netanyahu in whatever video is playing on your screen. It is a
**meme/parody tool** — the "Big Yahu" gag, not a surveillance product.

The detector is a small YOLO11n (5 MB) trained on publicly available photos
and campaign footage of a world leader, with two guard classes (`female`,
`other_male`) so it circles *him* and mostly nobody else.

## Demo

The 2015 Likud "Meet the Bibi Sitter" sketch, circled end to end:
**https://www.youtube.com/watch?v=P4jeMWcLuZ0**
(self-hosted copy: https://huggingface.co/spaces/minas2025/bibi-radar-demo)

## One command

```bash
python bibi_radar.py
```

That's it. First launch:

1. installs its own dependencies (`--user` pip, no venv needed),
2. downloads the model from the public Hugging Face repo
   [`minas2025/bibi-circleator`](https://huggingface.co/minas2025/bibi-circleator)
   into `~/.cache/bibi-radar`,
3. opens the radar window. Press **q** to quit.

Tuning: `python bibi_radar.py --conf 0.5` (default 0.45 — raise if it boxes
lookalikes, lower if it misses smiling young Bibi).

## Platforms

| OS | Status |
|---|---|
| Windows 10/11 | Supported: Python **3.10–3.13** x86_64 (openvino/opencv have no 3.14 wheels yet). Screen capture via `mss`. Inference is pure CPU. |
| Linux X11 | Supported via `mss`. |
| Linux Wayland | Supported via [`grim`](https://github.com/emersion/grim) (auto-detected). |
| macOS | Should work via `mss`; not personally tested — open an issue if it bites. |

## What it does with your data

Nothing. It reads the screen into RAM, draws circles, and forgets on the next
frame. No recording, no uploads, no face database, no network traffic after
the one-time model download.

## Honest limitations

- Trained on photos/frames of recent public appearances; extreme angles,
  heavy occlusion (masks, kippah-cam angles), and decades-old footage still
  miss sometimes.
- The `female` guard class is a heuristic gate, not a gender census — people
  who genuinely resemble him can still earn a circle. That's on the person
  next to him, not on you.
- Held-out test: bibi mAP50 0.992, other_male 0.958, female 0.838.

## How it works

Letterboxed 960px frame → OpenVINO CPU (≈12–35 ms/frame on a normal desktop)
→ per-anchor 3-channel argmax (the bibi score must *outvote* both guard
classes) → greedy NMS → circles. Model card and training notes live in the
HF repo.

## License

Code: MIT. Weights: **AGPL-3.0** — they derive from Ultralytics YOLO11, whose
license is AGPL, so sharing a modified detector publicly carries the same
terms. For meme overlays on a public figure's own footage, enjoy freely.
