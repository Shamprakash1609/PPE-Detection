# Construction PPE Detection

Three small object detectors trained to spot personal protective equipment on
construction sites — helmets, vests, gloves, boots, goggles — and, more importantly,
when they are **missing**.

Everything needed to run this is in the repository: the trained weights, the original
pretrained checkpoints, and the dataset. **Nothing downloads, and nothing needs training.**

---

## Quick start

```bash
git clone https://github.com/Shamprakash1609/PPE-Detection.git
cd PPE-Detection
pip install -r requirements.txt
```

Then open any notebook and run it top to bottom. Each takes about **30 seconds**.

> Clone it — do not download the ZIP from the GitHub web page. The ZIP can omit large
> files, which is the usual cause of "file not found" errors.

---

## The notebooks

| Notebook | What it does | Time |
|---|---|---|
| [`01_YOLO26n_PPE.ipynb`](01_YOLO26n_PPE.ipynb) | YOLO26 nano — 2.51 M params | ~30 s |
| [`02_YOLO11n_PPE.ipynb`](02_YOLO11n_PPE.ipynb) | YOLO11 nano — 2.59 M params | ~30 s |
| [`03_SSDLite_PPE.ipynb`](03_SSDLite_PPE.ipynb) | SSDLite + MobileNetV3 — 2.35 M params | ~30 s |
| [`04_Model_Comparison.ipynb`](04_Model_Comparison.ipynb) | All three, scored under one metric | ~40 s |

Each of 01–03 follows the same structure: check files → look at the data → load weights →
validate → **test predictions** → speed → **live camera**.

### Nothing trains by default

Every notebook has one switch near the top:

```python
TRAIN = False    # default — load the committed weights, no training
TRAIN = True     # retrain from scratch (about an hour on a GPU)
```

With `TRAIN = False` the notebook loads `trained_model/*.pt` and runs validation, test
predictions, benchmarks and the camera. The training curves and per-epoch tables still
render — they are read from `training_logs/`, which is committed too.

**You never have to train anything to see the results.**

---

## Live camera

The last section of notebooks 01–03 opens a window with live webcam predictions.
Missing-PPE detections are drawn in **red** and counted in the corner.

```python
ppe.live_camera(live, camera=0, conf=0.35)   # press Q or ESC to stop
```

It is commented out by default so a top-to-bottom run never blocks on a camera window.
Uncomment that one line and re-run the cell.

| Problem | Fix |
|---|---|
| `Could not open camera 0` | Run `ppe.list_cameras()` and use an index it reports |
| Window opens black | Another app holds the camera — close Teams, Zoom, OBS |
| Nothing happens in Colab | Expected: no display. Use local Jupyter or VS Code |
| Choppy | Pass `imgsz=320`, or use the SSDLite model |

Windows uses the DirectShow backend, which opens webcams far more reliably than the
default. macOS will ask for camera permission the first time.

---

## Results

Scored on the 141 held-out test images, all three under one metric:

| Model | Params | Size | mAP50 | mAP50-95 | FPS |
|---|---:|---:|---:|---:|---:|
| **YOLO11n** | 2.59 M | 5.2 MB | **0.5510** | **0.2763** | 130 |
| YOLO26n | 2.51 M | 5.2 MB | 0.5301 | 0.2631 | 118 |
| SSDLite | 2.35 M | 9.2 MB | 0.3951 | 0.1773 | 50 |

*(FPS measured on an Apple M2 Pro. A CUDA GPU is faster; CPU is 5–10x slower.)*

### What these models can and cannot do

They detect PPE that is **being worn** reasonably well. They are **not reliable** at
flagging PPE that is **missing**:

| Class | YOLO11n | YOLO26n | SSDLite | Training examples |
|---|---:|---:|---:|---:|
| vest | 0.520 | 0.496 | 0.421 | 1283 |
| Person | 0.510 | 0.509 | 0.458 | 1790 |
| helmet | 0.434 | 0.410 | 0.275 | 1357 |
| `no_helmet` | 0.163 | 0.091 | 0.088 | 400 |
| `no_gloves` | 0.072 | 0.063 | 0.016 | 442 |
| `no_goggle` | 0.045 | 0.060 | 0.028 | 337 |
| `no_boots` | 0.011 | 0.079 | 0.009 | 88 |

All three architectures fail on the same classes, in the same order, and that order
tracks how often each class appears in the training data. Three independent models
agreeing this closely is strong evidence the limit is **the dataset, not the
architecture** — a bigger model would not fix it.

**This is a demonstration, not a deployable safety system.** The classes a compliance
system exists to catch are the ones it detects worst.

What would actually help, in order of expected payoff:

1. Merge `no_helmet` / `no_goggle` / `no_gloves` / `no_boots` into one `missing_ppe`
   class, so their scarce examples pool.
2. Oversample images containing rare classes, or weight the loss by inverse frequency.
3. Label more missing-PPE examples — the real fix.

---

## Layout

```
PPE-Detection/
├── 01_YOLO26n_PPE.ipynb        training + testing + live camera
├── 02_YOLO11n_PPE.ipynb
├── 03_SSDLite_PPE.ipynb
├── 04_Model_Comparison.ipynb   all three under one metric
├── ppe_lib.py                  shared helpers (paths, models, drawing, camera)
├── requirements.txt
├── og_model/                   original COCO-pretrained checkpoints
├── trained_model/              the weights these notebooks use
├── training_logs/              per-epoch CSVs, so curves render without retraining
└── dataset/construction-ppe/   1416 images, train/val/test + YOLO labels
```

### `ppe_lib.py`

The notebooks stay readable because the fiddly parts live here:

```python
import ppe_lib as ppe

ppe.preflight()                    # verify every required file exists
det = ppe.Detector("yolo11n")      # load either model family
dets = det.predict("photo.jpg")    # same output shape regardless
ppe.show(det, "photo.jpg")         # draw inline
ppe.live_camera(det)               # live window
```

`Detector` hides the real differences between the two families: Ultralytics returns
original-image coordinates while torchvision returns them in a resized frame, and
SSDLite needs its classification head rebuilt before its weights will load at all.

---

## Requirements

Python 3.10–3.13. Works on Windows, macOS and Linux, with or without a GPU —
each notebook detects what is available and adjusts image size and batch size to match.

Dataset: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/),
11 classes, 1132 train / 143 val / 141 test. AGPL-3.0.
