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


def live_camera(detector: "Detector", camera: int = 0, conf: float = 0.35,
                imgsz: int | None = None, width: int = 1280, height: int = 720,
                window: str = "PPE Detection — press Q to quit",
                mirror: bool = True, max_seconds: float | None = None) -> None:
    """Open a window with live predictions from the webcam.

    Press Q or ESC to close. Runs in the kernel's own process, so the window belongs
    to whatever is hosting the notebook — in VS Code and Jupyter that is fine, but it
    will NOT appear in a browser-only environment such as Colab.
    """
    if imgsz is not None:
        original, detector.imgsz = detector.imgsz, imgsz

    with _quiet_stderr():
        cap = cv2.VideoCapture(camera, _capture_api())
    if not cap.isOpened():
        avail = list_cameras()
        print(f"Could not open camera {camera}.")
        print(f"Cameras that do open: {avail if avail else 'none found'}")
        if platform.system() == "Darwin":
            print("On macOS, grant camera access to the app hosting this kernel:")
            print("  System Settings > Privacy & Security > Camera")
        elif platform.system() == "Windows":
            print("On Windows, check Settings > Privacy > Camera, and close any other")
            print("app using the webcam (Teams and Zoom hold it exclusively).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Live: {detector.spec['label']} @ {detector.imgsz}px on {detector.backend.upper()}")
    print("Press Q or ESC in the video window to stop.")

    fps, prev, started, frames = 0.0, time.perf_counter(), time.perf_counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            dets = detector.predict(frame, conf=conf)
            out = draw(frame, dets)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            frames += 1

            missing = sum(1 for c in dets.labels if int(c) in MISSING_PPE)
            banner = f"{fps:5.1f} FPS | {len(dets)} detections"
            cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(out, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
            if missing:
                warn = f"MISSING PPE: {missing}"
                (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.putText(out, warn, (out.shape[1] - tw - 12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2)

            cv2.imshow(window, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break          # user clicked the X
            if max_seconds and (now - started) > max_seconds:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):     # macOS needs extra waitKey calls to actually close
            cv2.waitKey(1)
        if imgsz is not None:
            detector.imgsz = original
        print(f"Stopped after {frames} frames ({fps:.1f} FPS at the end).")


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
