"""
ppe_lib.py — shared helpers for the PPE detection notebooks.

Everything the notebooks need that isn't worth reading inline: locating the project,
checking that required files are actually present, loading either model family behind
one interface, drawing boxes, and running the live camera.

Works on Windows, macOS and Linux, with or without a GPU. No training code here —
this module is only for loading trained weights and running them.

    import ppe_lib as ppe

    ppe.preflight()                       # verify every required file exists
    det = ppe.Detector("yolo26n")         # load a trained model
    dets = det.predict("photo.jpg")       # -> Detections
    ppe.show(det, "photo.jpg")            # draw it in the notebook
    ppe.live_camera(det)                  # live window, press q to quit
"""

from __future__ import annotations

import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# MPS needs this before torch is imported; harmless everywhere else.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch

__all__ = [
    "CLASS_NAMES", "ROOT", "paths", "preflight", "get_device",
    "Detections", "Detector", "draw", "show", "live_camera", "MODELS",
    "bootstrap", "load_labels", "split_images", "list_cameras",
]

# ----------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------
CLASS_NAMES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots",
}
MISSING_PPE = {7, 8, 9, 10}          # the "not wearing it" classes — drawn in red
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------------
# Where am I?
# ----------------------------------------------------------------------------------
def _find_root() -> Path:
    """Locate the project root from wherever the kernel happened to start.

    Notebook working directories are unreliable: VS Code, Jupyter Lab and `jupyter
    nbconvert` each pick a different one, and on Windows people often open the parent
    folder. Rather than assume, walk up from both this file and the cwd looking for
    the folder that actually contains the weights.
    """
    marker = "trained_model"
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / marker).is_dir():
                return candidate
    # Fall back to this file's folder — preflight() will report what's missing.
    return Path(__file__).resolve().parent


ROOT = _find_root()


@dataclass(frozen=True)
class Paths:
    root: Path
    og_model: Path
    trained_model: Path
    dataset: Path
    images: Path
    labels: Path
    outputs: Path

    def split(self, name: str) -> Path:
        return self.images / name


def _make_paths(root: Path) -> Paths:
    ds = root / "dataset" / "construction-ppe"
    return Paths(
        root=root,
        og_model=root / "og_model",
        trained_model=root / "trained_model",
        dataset=ds,
        images=ds / "images",
        labels=ds / "labels",
        outputs=root / "outputs",
    )


paths = _make_paths(ROOT)


# ----------------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------------
MODELS = {
    "yolo26n": {"family": "yolo", "weights": "yolo26n_best.pt", "imgsz": 640,
                "label": "YOLO26 nano"},
    "yolo11n": {"family": "yolo", "weights": "yolo11n_best.pt", "imgsz": 640,
                "label": "YOLO11 nano"},
    "yolo11s": {"family": "yolo", "weights": "yolo11s_best.pt", "imgsz": 640,
                "label": "YOLO11 small"},
    "ssdlite": {"family": "ssd", "weights": "ssdlite_best.pt", "imgsz": 320,
                "label": "SSDLite320-MobileNetV3"},
}


# ----------------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------------
def preflight(require_train_split: bool = False, verbose: bool = True) -> bool:
    """Check every file the notebooks need, and say exactly what is missing.

    The point is to fail in one readable place at the top of the notebook rather than
    with a FileNotFoundError forty cells later. Returns True when everything needed
    for inference is present.
    """
    rows, ok = [], True

    def check(label, path, needed=True, extra=""):
        nonlocal ok
        exists = path.exists()
        if needed and not exists:
            ok = False
        rows.append((label, "OK" if exists else ("MISSING" if needed else "absent"),
                     str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                     extra))
        return exists

    for key, spec in MODELS.items():
        w = paths.trained_model / spec["weights"]
        size = f"{w.stat().st_size/1024**2:.1f} MB" if w.exists() else ""
        check(f"weights: {key}", w, needed=True, extra=size)

    splits = ["test", "val"] + (["train"] if require_train_split else [])
    for s in splits:
        d = paths.split(s)
        n = sum(1 for p in d.glob("*") if p.suffix.lower() in IMG_EXT) if d.is_dir() else 0
        check(f"images/{s}", d, needed=True, extra=f"{n} images")
        ld = paths.labels / s
        nl = len(list(ld.glob("*.txt"))) if ld.is_dir() else 0
        check(f"labels/{s}", ld, needed=True, extra=f"{nl} files")

    if verbose:
        w1 = max(len(r[0]) for r in rows) + 2
        w2 = 9
        print("=" * 78)
        print(f"  PREFLIGHT — project root: {ROOT}")
        print("=" * 78)
        for label, status, where, extra in rows:
            mark = "v" if status == "OK" else ("X" if status == "MISSING" else "-")
            print(f"  {mark} {label:<{w1}}{status:<{w2}}{extra:<12}{where}")
        print("=" * 78)
        if ok:
            print("  All required files present. Everything below will run.")
        else:
            print("  MISSING FILES — see the X rows above.")
            print()
            print("  Most likely cause: the repository was downloaded as a ZIP without")
            print("  the large files, or cloned partially. Re-clone with:")
            print("      git clone https://github.com/Shamprakash1609/PPE-Detection.git")
        print()
    return ok


def _ssd_extras(model_name: str, imgsz: int, det) -> dict:
    """The SSDLite notebook's own names — it has a hand-written training loop.

    Notebooks 01/02 lean on Ultralytics, so their later sections need only a model
    and some paths. Notebook 03 builds its own Dataset, loaders and history list, so
    running it from the middle needs those rebuilt too.
    """
    import pandas as _pd
    import torch as _torch
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
    from torchmetrics.detection import MeanAveragePrecision as _MAP

    num_classes = len(CLASS_NAMES) + 1
    idx_to_name = {i + 1: n for i, n in CLASS_NAMES.items()}

    class PPEDataset(_Dataset):
        def __init__(self, root, split, size=imgsz, train=False):
            self.img_dir = Path(root) / "images" / split
            self.lbl_dir = Path(root) / "labels" / split
            self.imgsz, self.train = size, train
            self.items = sorted(q for q in self.img_dir.glob("*")
                                if q.suffix.lower() in IMG_EXT)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path = self.items[i]
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (self.imgsz, self.imgsz))
            boxes, labels = [], []
            lbl = self.lbl_dir / f"{path.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    x1 = max(0.0, (cx - bw / 2) * self.imgsz)
                    y1 = max(0.0, (cy - bh / 2) * self.imgsz)
                    x2 = min(float(self.imgsz), (cx + bw / 2) * self.imgsz)
                    y2 = min(float(self.imgsz), (cy + bh / 2) * self.imgsz)
                    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    labels.append(c + 1)
            t = _torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return t, {
                "boxes": _torch.tensor(boxes, dtype=_torch.float32) if boxes
                         else _torch.zeros((0, 4), dtype=_torch.float32),
                "labels": _torch.tensor(labels, dtype=_torch.int64) if labels
                          else _torch.zeros((0,), dtype=_torch.int64),
                "image_id": _torch.tensor([i]),
                "orig_size": _torch.tensor([h0, w0]),
                "path": str(path),
            }

    def collate(batch):
        return tuple(zip(*batch))

    train_ds = PPEDataset(paths.dataset, "train", train=True)
    val_ds   = PPEDataset(paths.dataset, "val")
    test_ds  = PPEDataset(paths.dataset, "test")

    log = paths.root / "training_logs" / f"{model_name}_results.csv"
    history = _pd.read_csv(log).to_dict(orient="records") if log.exists() else []
    best = ({"map": max(h["mAP50-95"] for h in history),
             "epoch": int(max(history, key=lambda h: h["mAP50-95"])["epoch"])}
            if history else {"map": -1.0, "epoch": 0})

    return {
        "model": det.model, "NUM_CLASSES": num_classes, "IDX_TO_NAME": idx_to_name,
        "PPEDataset": PPEDataset, "collate": collate,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "val_loader": _DataLoader(val_ds, batch_size=8, shuffle=False,
                                  num_workers=0, collate_fn=collate),
        "train_loader": _DataLoader(train_ds, batch_size=8, shuffle=True,
                                    num_workers=0, collate_fn=collate),
        "metric": _MAP(box_format="xyxy", iou_type="bbox", class_metrics=True),
        "history": history, "best": best,
        # the epoch table in section 11 builds `hist`; the recap in section 16
        # uses it, so provide it for anyone starting from section 12
        "hist": _pd.DataFrame(history),
        "PRED_DIR": paths.root / "predictions" / model_name,
        "CONF": 0.30, "SEED": 0, "LR": 1e-3,
    }


def bootstrap(model_name: str, verbose: bool = True) -> dict:
    """Rebuild everything the validation / prediction / export sections need.

    Those sections depend only on the trained weights and the dataset, but they
    reference names created by the setup and training cells above them. Restart the
    kernel, or jump straight to the testing section, and you get a cascade of
    NameErrors that look alarming and have nothing to do with the model.

    This reconstructs that state from disk. Feed the result through
    `globals().setdefault(...)` so anything the earlier cells already defined is
    left exactly as it was.
    """
    import json as _json, platform as _platform, shutil as _shutil, sys as _sys
    import time as _time
    from collections import Counter as _Counter
    import numpy as _np, pandas as _pd
    import matplotlib.pyplot as _plt

    spec = MODELS[model_name]
    dev, backend = get_device(verbose=False)
    gpu = backend != "cpu"

    weights = paths.trained_model / spec["weights"]
    if not weights.exists():
        raise FileNotFoundError(chr(10).join([
            "=" * 70,
            f"  Trained weights not found for '{model_name}':",
            f"    {weights}",
            "",
            "  Re-clone the repository — the weights are committed there.",
            "=" * 70]))

    # class frequencies in the training split, used by the per-class charts
    counts = _Counter()
    train_lbl = paths.labels / "train"
    if train_lbl.is_dir():
        for f in train_lbl.glob("*.txt"):
            for line in f.read_text().strip().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1

    det = None
    if spec["family"] == "yolo":
        from ultralytics import YOLO as _CLS
        model = _CLS(str(weights))
        n_params = sum(q.numel() for q in model.model.parameters())
    else:
        det = Detector(model_name, device=dev, backend=backend, verbose=False)
        _CLS, model, n_params = type(det.model), det.model, det.n_params

    run_dir = paths.root / "runs" / model_name
    logs = paths.root / "training_logs"

    def results_csv():
        for cand in (run_dir / "results.csv", logs / f"{model_name}_results.csv"):
            if cand.exists():
                return cand
        return None

    # Folders the later cells write into. A fresh clone has none of them, and
    # cells like the exporter do shutil.copy2 into EXPORT_DIR without creating
    # it, so make them here rather than letting the first write crash.
    for _d in (paths.root / "runs" / model_name,
               paths.root / "predictions" / model_name,
               paths.root / "exports", paths.outputs):
        _d.mkdir(parents=True, exist_ok=True)

    ctx = {
        # modules the cells use
        "json": _json, "sys": _sys, "time": _time, "platform": _platform,
        "shutil": _shutil, "Path": Path, "Counter": _Counter,
        "np": _np, "pd": _pd, "plt": _plt, "cv2": cv2, "torch": torch,
        # project layout
        "PROJECT": paths.root, "DATA_ROOT": paths.dataset,
        "DATA_YAML": paths.dataset.parent / "construction-ppe.yaml",
        "RUNS_DIR": paths.root / "runs", "PRED_ROOT": paths.root / "predictions",
        "TRAINED_DIR": paths.trained_model, "EXPORT_DIR": paths.root / "exports",
        "LOGS_DIR": logs, "SAVE_DIR": run_dir,
        # runtime
        "DEVICE": str(dev) if spec["family"] == "yolo" else dev,
        "GPU_AVAILABLE": gpu, "GPU_TYPE": backend if gpu else None,
        "GPU_NAME": backend.upper() if gpu else "none",
        "HOST": {"os": f"{_platform.system()} {_platform.release()}",
                 "arch": _platform.machine(), "cores": os.cpu_count()},
        "BACKEND": backend,
        # config
        "RUN_NAME": model_name, "MODEL_NAME": spec["weights"],
        "MODEL_CLS": _CLS, "BEST": weights, "OG_WEIGHTS": paths.og_model / spec["weights"],
        "IMGSZ": spec["imgsz"], "BATCH": 16 if gpu else 4,
        "WORKERS": 0, "AMP": backend == "cuda", "EPOCHS": 0,
        "BENCH_RUNS": 20 if gpu else 5,
        "BENCH_SIZES": (640, 512, 416, 320) if gpu else (416, 320),
        # names the training cells would have produced
        "CLASS_NAMES": CLASS_NAMES, "IMG_EXT": IMG_EXT, "counts": counts,
        "n_params": n_params, "elapsed": 0.0, "epoch_times": [],
        "guard": {"cooldowns": 0, "baseline": None, "paused": 0.0},
        "saved": {k: str(paths.trained_model / f"{model_name}_{k}.pt")
                  for k in ("best", "last")
                  if (paths.trained_model / f"{model_name}_{k}.pt").exists()},
        "results_csv": results_csv,
        "best_model": model,
    }

    if spec["family"] == "ssd":
        ctx.update(_ssd_extras(model_name, spec["imgsz"], det))

    if verbose:
        log = results_csv()
        print(f"Section context rebuilt for '{model_name}':")
        print(f"  weights   : {weights.name}  ({weights.stat().st_size/1024**2:.2f} MB)")
        print(f"  device    : {backend.upper()}")
        print(f"  imgsz     : {spec['imgsz']}")
        print(f"  epoch log : {log.name if log else 'not found'}")
        print(f"  train class counts loaded: {sum(counts.values())} boxes")
    return ctx


# ----------------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------------
def get_device(verbose: bool = True):
    """Pick CUDA, then Apple Metal, then CPU. Returns (torch.device, backend_name)."""
    if torch.cuda.is_available():
        dev, backend = torch.device("cuda:0"), "cuda"
        detail = f"{torch.cuda.get_device_name(0)}, " \
                 f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB"
    elif bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        dev, backend = torch.device("mps"), "mps"
        detail = "Apple Silicon GPU (Metal)"
    else:
        dev, backend = torch.device("cpu"), "cpu"
        detail = f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores"

    if verbose:
        print(f"Device : {backend.upper()}  ({detail})")
        if backend == "cpu":
            print("         No GPU found. Inference still works — expect roughly")
            print("         5-15 FPS instead of 50-100. Live camera will be choppy;")
            print("         pass imgsz=320 to live_camera() to help.")
    return dev, backend


def _sync(backend: str) -> None:
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------------
# Detections
# ----------------------------------------------------------------------------------
@dataclass
class Detections:
    """Model output in one shape, whatever produced it.

    boxes  : (N, 4) float array of xyxy in ORIGINAL image pixels
    scores : (N,) float array
    labels : (N,) int array of 0-indexed class ids matching CLASS_NAMES
    """
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def names(self) -> list[str]:
        return [CLASS_NAMES.get(int(c), str(c)) for c in self.labels]

    def summary(self) -> str:
        if not len(self):
            return "nothing detected"
        from collections import Counter
        c = Counter(self.names())
        return ", ".join(f"{k} x{v}" for k, v in c.most_common())

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "class": self.names(),
            "confidence": np.round(self.scores, 3),
            "x1": np.round(self.boxes[:, 0]).astype(int) if len(self) else [],
            "y1": np.round(self.boxes[:, 1]).astype(int) if len(self) else [],
            "x2": np.round(self.boxes[:, 2]).astype(int) if len(self) else [],
            "y2": np.round(self.boxes[:, 3]).astype(int) if len(self) else [],
        }).sort_values("confidence", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Detector — one interface over two very different model families
# ----------------------------------------------------------------------------------
class Detector:
    """Load a trained model and run it, regardless of which family it came from.

    YOLO models come from Ultralytics and already return original-image coordinates.
    SSDLite comes from torchvision, needs its head rebuilt before the weights load,
    and returns boxes in the resized 320x320 frame which must be scaled back. All of
    that is hidden here so the notebooks can just call .predict().
    """

    def __init__(self, name: str, device=None, backend: str | None = None,
                 conf: float = 0.30, verbose: bool = True):
        if name not in MODELS:
            raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS)}")
        self.name = name
        self.spec = MODELS[name]
        self.conf = conf
        self.imgsz = self.spec["imgsz"]

        if device is None:
            device, backend = get_device(verbose=False)
        self.device, self.backend = device, backend or str(device).split(":")[0]

        w = paths.trained_model / self.spec["weights"]
        if not w.exists():
            raise FileNotFoundError(
                f"\nTrained weights not found:\n  {w}\n\n"
                f"Run ppe_lib.preflight() to see what else is missing, or re-clone "
                f"the repository."
            )
        self.weights_path = w

        if self.spec["family"] == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(str(w))
            self.n_params = sum(p.numel() for p in self.model.model.parameters())
        else:
            self.model = self._build_ssdlite(w)
            self.n_params = sum(p.numel() for p in self.model.parameters())

        if verbose:
            print(f"{self.spec['label']:<26} loaded  "
                  f"{self.n_params/1e6:.2f} M params  "
                  f"{w.stat().st_size/1024**2:.1f} MB  @ {self.imgsz}px")

    def _build_ssdlite(self, weights: Path):
        import torch.nn as nn
        from functools import partial
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        from torchvision.models.detection import _utils as det_utils

        m = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        in_ch = det_utils.retrieve_out_channels(m.backbone, (self.imgsz, self.imgsz))
        n_anchors = m.anchor_generator.num_anchors_per_location()
        # The saved state dict has a 12-class head (11 PPE + background); a fresh model
        # has COCO's 91. Rebuild before loading or it fails on a shape mismatch.
        m.head.classification_head = SSDLiteClassificationHead(
            in_ch, n_anchors, len(CLASS_NAMES) + 1,
            partial(nn.BatchNorm2d, eps=0.001, momentum=0.03))
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        return m.to(self.device).eval()

    # -- inference ------------------------------------------------------------
    def predict(self, image, conf: float | None = None) -> Detections:
        """Run on a path or a BGR numpy frame. Returns original-image coordinates."""
        conf = self.conf if conf is None else conf

        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        else:
            frame = image
        h, w = frame.shape[:2]

        if self.spec["family"] == "yolo":
            r = self.model.predict(frame, imgsz=self.imgsz, device=str(self.device),
                                   conf=conf, verbose=False)[0]
            return Detections(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.conf.cpu().numpy(),
                              r.boxes.cls.cpu().numpy().astype(np.int64))

        rgb = cv2.cvtColor(cv2.resize(frame, (self.imgsz, self.imgsz)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.).to(self.device)
        with torch.no_grad():
            o = self.model([t])[0]
        keep = o["scores"] >= conf
        b = o["boxes"][keep].detach().cpu().numpy()
        if len(b):
            # scale the 320x320 frame back to the original image
            b = b * np.array([w / self.imgsz, h / self.imgsz,
                              w / self.imgsz, h / self.imgsz], np.float32)
        # torchvision reserves 0 for background, so shift to match CLASS_NAMES
        return Detections(b, o["scores"][keep].detach().cpu().numpy(),
                          o["labels"][keep].detach().cpu().numpy().astype(np.int64) - 1)

    def benchmark(self, images, runs: int = 20, warmup: int = 3) -> dict:
        images = [str(p) for p in images]
        frames = [cv2.imread(p) for p in images]
        for i in range(warmup):
            self.predict(frames[i % len(frames)])
        _sync(self.backend)
        ts = []
        for i in range(runs):
            t0 = time.perf_counter()
            self.predict(frames[i % len(frames)])
            _sync(self.backend)
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        return {"mean_ms": float(ts.mean()), "median_ms": float(np.median(ts)),
                "fps": float(1000 / ts.mean())}


# ----------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------
def _colour(cls_id: int) -> tuple[int, int, int]:
    """BGR. Red for missing-PPE classes, green for worn PPE, blue for person/none."""
    if cls_id in MISSING_PPE:
        return (60, 60, 220)
    if cls_id in (5, 6):
        return (200, 140, 40)
    return (70, 180, 70)


def draw(frame: np.ndarray, dets: Detections, show_conf: bool = True) -> np.ndarray:
    """Draw boxes on a BGR frame and return a new BGR frame."""
    out = frame.copy()
    scale = max(0.5, min(1.4, max(out.shape[:2]) / 900))
    for (x1, y1, x2, y2), s, c in zip(dets.boxes, dets.scores, dets.labels):
        c = int(c)
        if not (0 <= c < len(CLASS_NAMES)):
            continue
        col = _colour(c)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, col, max(2, int(2 * scale)))
        text = f"{CLASS_NAMES[c]} {s:.2f}" if show_conf else CLASS_NAMES[c]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        ty = max(p1[1] - 4, th + 4)
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty + 2), col, -1)
        cv2.putText(out, text, (p1[0] + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (255, 255, 255), max(1, int(scale)))
    return out


def show(detector: "Detector", image, conf: float | None = None, side_by_side: bool = True,
         figsize=(14, 6)):
    """Predict and display inline. Returns the Detections."""
    import matplotlib.pyplot as plt
    frame = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    dets = detector.predict(frame, conf=conf)
    annotated = draw(frame, dets)

    if side_by_side:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Input", fontsize=10)
        ax[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"{detector.spec['label']} — {dets.summary()}", fontsize=10)
        for a in ax:
            a.axis("off")
    else:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f"{detector.spec['label']} — {dets.summary()}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    return dets


# ----------------------------------------------------------------------------------
# Live camera
# ----------------------------------------------------------------------------------
class _quiet_stderr:
    """Silence native (C-level) stderr for the duration of a block.

    Probing a missing camera makes AVFoundation and DirectShow write directly to
    file descriptor 2, below anything Python or OpenCV's logger can intercept, so
    try/except and cv2 log levels both fail to stop it. Swapping the descriptor is
    the only thing that reliably works.
    """

    def __enter__(self):
        self._saved = None
        try:
            sys.stderr.flush()
            self._saved = os.dup(2)
            self._null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._null, 2)
        except Exception:
            self._saved = None
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            try:
                sys.stderr.flush()
                os.dup2(self._saved, 2)
                os.close(self._saved)
                os.close(self._null)
            except Exception:
                pass
        return False


def list_cameras(max_index: int = 4) -> list[int]:
    """Indices that actually open. Useful when 0 is a virtual camera.

    Probing a camera that isn't there makes OpenCV print several lines of native
    error text straight to stderr — alarming to read and impossible to catch with
    try/except, since it comes from C++ rather than Python. Turn its log level down
    for the duration of the probe.
    """
    prev = None
    try:
        prev = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    found = []
    try:
        with _quiet_stderr():
            for i in range(max_index):
                cap = cv2.VideoCapture(i, _capture_api())
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        found.append(i)
                cap.release()
    finally:
        if prev is not None:
            try:
                cv2.utils.logging.setLogLevel(prev)
            except Exception:
                pass
    return found


def _capture_api() -> int:
    """Pick the capture backend that behaves on each OS.

    On Windows the default backend is often MSMF, which can take many seconds to open
    a webcam and sometimes returns empty frames. DirectShow is markedly more reliable.
    """
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class _FrameGrabber:
    """Read frames in a background thread and keep only the newest one.

    This is the fix for webcam lag, and it is worth understanding why.

    `cap.read()` blocks — measured at ~35 ms on this machine, so a serial
    read-infer-draw loop is capped near 28 FPS no matter how fast the model is
    (inference is ~6 ms). Worse, the driver queues frames while inference runs, and
    `read()` returns the OLDEST queued frame, so displayed latency grows the longer
    you watch. `CAP_PROP_BUFFERSIZE = 1` is the usual remedy but many backends
    silently reject it (AVFoundation returns False and leaves it at -1).

    Reading continuously in a thread drains that queue as fast as the camera fills
    it, and the main loop always takes the most recent frame. Capture then overlaps
    inference instead of serialising with it, and latency stays flat.
    """

    def __init__(self, src: int, api: int, width: int, height: int):
        import threading
        with _quiet_stderr():
            self.cap = cv2.VideoCapture(src, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {src} would not open")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # honoured on some backends only

        self._frame = None
        self._seq = 0
        self._stop = False
        self._lock = threading.Lock()
        self._fps = 0.0
        ok, first = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError(f"camera {src} opened but returned no frames")
        self._frame, self._seq = first, 1
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        prev = time.perf_counter()
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            dt = now - prev
            prev = now
            with self._lock:
                # publish by reference — cap.read() allocates a fresh array each
                # time, so no copy is needed and none is made
                self._frame = f
                self._seq += 1
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

    def latest(self):
        """Newest frame and its sequence number. Never blocks on the camera."""
        with self._lock:
            return self._frame, self._seq

    @property
    def capture_fps(self) -> float:
        with self._lock:
            return self._fps

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


_HELP = [
    "Q / ESC   quit",
    "SPACE     pause",
    "+ / -     confidence",
    "[ / ]     input size",
    "M         next model",
    "B         boxes on/off",
    "S         save snapshot",
    "H         hide this help",
]


def _hud(img, lines, origin=(10, 10), alpha=0.55):
    """Translucent panel so text stays readable over any footage."""
    x, y = origin
    pad, lh = 8, 20
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines) + pad * 2
    h = lh * len(lines) + pad
    panel = img[y:y + h, x:x + w]
    if panel.size:
        img[y:y + h, x:x + w] = cv2.addWeighted(
            panel, 1 - alpha, np.full_like(panel, 30), alpha, 0)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + pad, y + pad + lh * i + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


def live_camera(detector, camera: int = 0, conf: float = 0.35,
                imgsz: int | None = None, width: int = 1280, height: int = 720,
                window: str = "PPE Detection",
                mirror: bool = True, max_seconds: float | None = None,
                models: dict | None = None, show_help: bool = True) -> None:
    """Live webcam detection in a window, with interactive controls.

    Capture runs in its own thread so it never blocks inference — see
    _FrameGrabber for why that matters more than model speed here.

    Keys
        Q / ESC   quit              SPACE   pause
        + / -     confidence        [ / ]   input size
        M         next model        B       toggle boxes
        S         save a snapshot   H       toggle the help panel

    `models` may be a {name: Detector} dict to enable model switching with M.
    """
    sizes = [320, 416, 512, 640]
    if imgsz is not None:
        detector.imgsz = imgsz
    if detector.imgsz not in sizes:
        sizes = sorted(set(sizes + [detector.imgsz]))

    pool = list(models.items()) if models else [(detector.name, detector)]
    idx = next((i for i, (n, _) in enumerate(pool) if n == detector.name), 0)
    det = pool[idx][1]
    det.imgsz = detector.imgsz

    try:
        grab = _FrameGrabber(camera, _capture_api(), width, height)
    except RuntimeError as e:
        print(f"Could not start camera {camera}: {e}")
        avail = list_cameras()
        print(f"Cameras that do open: {avail if avail else 'none found'}")
        if platform.system() == "Darwin":
            print("macOS: System Settings > Privacy & Security > Camera, then allow")
            print("the app hosting this kernel (VS Code / Terminal).")
        elif platform.system() == "Windows":
            print("Windows: Settings > Privacy > Camera. Close Teams/Zoom first —")
            print("they hold the webcam exclusively.")
        return

    snaps = paths.outputs / "snapshots"
    print(f"Live: {det.spec['label']} @ {det.imgsz}px on {det.backend.upper()}")
    print("Q quit | SPACE pause | +/- conf | [ ] size | M model | B boxes | S save | H help")

    dets = Detections()
    paused = False
    boxes_on = True
    dirty = True          # force one draw on entry
    pending_key = None    # key captured during an idle wait
    infer_ms, disp_fps = 0.0, 0.0
    last_seq, shots, frames = -1, 0, 0
    prev = time.perf_counter()
    started = prev

    try:
        while True:
            frame, seq = grab.latest()
            if frame is None:
                time.sleep(0.005)
                continue

            fresh = seq != last_seq
            if not paused and fresh:
                last_seq = seq
                t0 = time.perf_counter()
                dets = det.predict(frame, conf=conf)
                infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t0) * 1000
            elif not dirty:
                # No new frame and nothing changed on screen. Redrawing would just
                # burn CPU — the threaded grabber lets this loop spin far faster
                # than the camera produces frames. Still call waitKey so the window
                # stays responsive to keys and to being closed.
                k = cv2.waitKey(3) & 0xFF
                if k != 255:
                    pending_key = k
                else:
                    try:
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                    if max_seconds and (time.perf_counter() - started) > max_seconds:
                        break
                    continue

            view = frame.copy()
            if mirror:
                view = cv2.flip(view, 1)
            if boxes_on and len(dets):
                d = dets
                if mirror:                      # mirror the boxes to match the image
                    w = view.shape[1]
                    b = d.boxes.copy()
                    b[:, [0, 2]] = w - d.boxes[:, [2, 0]]
                    d = Detections(b, d.scores, d.labels)
                view = draw(view, d)

            now = time.perf_counter()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            frames += 1

            missing = sum(1 for c in dets.labels if int(c) in MISSING_PPE)
            status = [
                f"{det.spec['label']}  {det.imgsz}px  conf {conf:.2f}",
                f"display {disp_fps:5.1f} FPS | camera {grab.capture_fps:5.1f} FPS | "
                f"infer {infer_ms:4.1f} ms",
                f"{len(dets)} detections" + ("  [PAUSED]" if paused else ""),
            ]
            _hud(view, status)
            if show_help:
                _hud(view, _HELP, origin=(10, view.shape[0] - 20 * len(_HELP) - 18))
            if missing:
                txt = f"MISSING PPE x{missing}"
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(view, txt, (view.shape[1] - tw - 14, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 240), 2, cv2.LINE_AA)

            cv2.imshow(window, view)
            dirty = False
            k = pending_key if pending_key is not None else (cv2.waitKey(1) & 0xFF)
            pending_key = None

            if k != 255:
                dirty = True      # a key changed something, so redraw next pass

            if k in (ord("q"), ord("Q"), 27):
                break
            elif k == 32:
                paused = not paused
            elif k in (ord("+"), ord("=")):
                conf = min(0.95, round(conf + 0.05, 2))
            elif k in (ord("-"), ord("_")):
                conf = max(0.05, round(conf - 0.05, 2))
            elif k == ord("]"):
                det.imgsz = sizes[min(sizes.index(det.imgsz) + 1, len(sizes) - 1)]
            elif k == ord("["):
                det.imgsz = sizes[max(sizes.index(det.imgsz) - 1, 0)]
            elif k in (ord("m"), ord("M")) and len(pool) > 1:
                keep = det.imgsz
                idx = (idx + 1) % len(pool)
                det = pool[idx][1]
                det.imgsz = keep if keep in sizes else det.imgsz
            elif k in (ord("b"), ord("B")):
                boxes_on = not boxes_on
            elif k in (ord("h"), ord("H")):
                show_help = not show_help
            elif k in (ord("s"), ord("S")):
                snaps.mkdir(parents=True, exist_ok=True)
                shots += 1
                out = snaps / f"snap_{shots:03d}.jpg"
                cv2.imwrite(str(out), view)
                print(f"saved {out}")

            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if max_seconds and (now - started) > max_seconds:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        grab.release()
        cv2.destroyAllWindows()
        for _ in range(5):          # macOS needs the extra waitKey to actually close
            cv2.waitKey(1)
        print(f"Stopped after {frames} frames — {disp_fps:.1f} FPS display, "
              f"{grab.capture_fps:.1f} FPS camera, {infer_ms:.1f} ms inference."
              + (f" {shots} snapshot(s) in {snaps}" if shots else ""))


# ----------------------------------------------------------------------------------
# Ground truth (for the validation section)
# ----------------------------------------------------------------------------------
def load_labels(image_path: Path, split: str) -> Detections:
    """Read the YOLO .txt label for an image as absolute xyxy in original pixels."""
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    h, w = frame.shape[:2]
    lbl = paths.labels / split / f"{image_path.stem}.txt"
    boxes, labels = [], []
    if lbl.exists():
        for line in lbl.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(c)
    return Detections(
        np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32),
        np.ones(len(boxes), np.float32),
        np.array(labels, np.int64) if labels else np.zeros((0,), np.int64))


def split_images(split: str) -> list[Path]:
    d = paths.split(split)
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT) if d.is_dir() else []
