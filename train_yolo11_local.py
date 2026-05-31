"""
YOLO11 Object Detection - Training Script (Local / Laptop)
Converted from Google Colab notebook to standalone Python script.

Requirements:
    pip install "ultralytics<=8.3.40" supervision roboflow opencv-python matplotlib

Usage:
    python train_yolo11_local.py
"""

import os
import sys
import glob
import random
import subprocess

# ─────────────────────────────────────────────
# 0. CEK SPESIFIKASI PC (GPU / CPU)
# ─────────────────────────────────────────────
print("=" * 60)
print("  YOLO11 Training - Local Mode")
print("=" * 60)

def check_device():
    """Cek apakah GPU (CUDA) tersedia, fallback ke CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n✅  GPU Ditemukan: {gpu_name}")
            print(f"    VRAM           : {vram:.1f} GB")
            print(f"    CUDA Version   : {torch.version.cuda}")
            device = "cuda"
        else:
            print("\n⚠️  GPU tidak ditemukan atau CUDA tidak tersedia.")
            print("    Training akan menggunakan CPU (lebih lambat).")
            device = "cpu"
    except ImportError:
        print("\n⚠️  PyTorch belum terinstall. Akan diinstall bersama ultralytics.")
        device = "cpu"

    print(f"\n→  Device yang digunakan: [{device.upper()}]")
    print("=" * 60)
    return device

DEVICE = check_device()

# ─────────────────────────────────────────────
# 1. INSTALL DEPENDENCIES (jika belum ada)
# ─────────────────────────────────────────────
print("\n[1/5] Mengecek & menginstall dependencies...")

try:
    import ultralytics
    print("    ultralytics sudah terinstall.")
except ImportError:
    print("    Menginstall ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics<=8.3.40", "supervision", "roboflow"])

import ultralytics
ultralytics.checks()

# Nonaktifkan telemetry ultralytics
os.system("yolo settings sync=False")

# ─────────────────────────────────────────────
# 2. SETUP HOME DIRECTORY
# ─────────────────────────────────────────────
HOME = os.getcwd()
print(f"\n[2/5] HOME directory: {HOME}")

os.makedirs(os.path.join(HOME, "datasets"), exist_ok=True)

# ─────────────────────────────────────────────
# 3. DOWNLOAD DATASET DARI ROBOFLOW
# ─────────────────────────────────────────────
print("\n[3/5] Mendownload dataset dari Roboflow...")

os.chdir(os.path.join(HOME, "datasets"))

from roboflow import Roboflow

ROBOFLOW_API_KEY = "d6aoFexnRf14ZIHN4LUb"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("ai-world-00rg5").project("ta-19")
version = project.version(3)
dataset = version.download("yolov11")

print(f"\n    Dataset tersimpan di: {dataset.location}")

# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
print("\n[4/5] Memulai training YOLO11...")

os.chdir(HOME)

from ultralytics import YOLO

# Pilih model size berdasarkan device
# GPU  → yolo11s.pt (lebih akurat)
# CPU  → yolo11n.pt (paling ringan, lebih cepat di CPU)
if DEVICE == "cuda":
    MODEL_SIZE = "yolo11s.pt"
    EPOCHS     = 50
    IMGSZ      = 640
    BATCH      = 16
    print(f"    Mode GPU → model: {MODEL_SIZE}, epochs: {EPOCHS}, imgsz: {IMGSZ}, batch: {BATCH}")
else:
    MODEL_SIZE = "yolo11n.pt"
    EPOCHS     = 100
    IMGSZ      = 640
    BATCH      = 8
    print(f"    Mode CPU → model: {MODEL_SIZE}, epochs: {EPOCHS}, imgsz: {IMGSZ}, batch: {BATCH}")
    print("    ⚠️  Training di CPU bisa memakan waktu lama. Mohon bersabar.")

model = YOLO(MODEL_SIZE)

results = model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    plots=True,
    project=os.path.join(HOME, "runs", "detect"),
    name="train",
    exist_ok=True,
)

print("\n✅  Training selesai!")
print(f"    Weights tersimpan di: {HOME}/runs/detect/train/weights/best.pt")

# ─────────────────────────────────────────────
# 5. VALIDASI MODEL
# ─────────────────────────────────────────────
print("\n[5/5] Validasi model terbaik...")

best_weights = os.path.join(HOME, "runs", "detect", "train", "weights", "best.pt")
val_model = YOLO(best_weights)

val_results = val_model.val(
    data=os.path.join(dataset.location, "data.yaml"),
    device=DEVICE,
)

# ─────────────────────────────────────────────
# 6. INFERENCE PADA TEST IMAGES
# ─────────────────────────────────────────────
print("\n[6/6] Inferensi pada test images...")

import cv2
import matplotlib.pyplot as plt
import supervision as sv

test_images_dir = os.path.join(dataset.location, "test", "images")

if not os.path.exists(test_images_dir):
    print(f"    ⚠️  Folder test images tidak ditemukan: {test_images_dir}")
else:
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_images_dir, "*.png"))

    if not test_images:
        print("    ⚠️  Tidak ada gambar test ditemukan.")
    else:
        sample = random.sample(test_images, min(4, len(test_images)))

        infer_model = YOLO(best_weights)
        box_annotator   = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

        fig, axes = plt.subplots(1, len(sample), figsize=(6 * len(sample), 6))
        if len(sample) == 1:
            axes = [axes]

        for ax, img_path in zip(axes, sample):
            image_bgr = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            result = infer_model.predict(image_bgr, conf=0.25)[0]
            detections = sv.Detections.from_ultralytics(result)

            annotated = image_rgb.copy()
            annotated = box_annotator.annotate(annotated, detections=detections)
            annotated = label_annotator.annotate(annotated, detections=detections)

            ax.imshow(annotated)
            ax.set_title(os.path.basename(img_path), fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        output_fig = os.path.join(HOME, "runs", "detect", "inference_preview.png")
        plt.savefig(output_fig, dpi=150, bbox_inches="tight")
        print(f"    Preview inferensi disimpan di: {output_fig}")
        plt.show()

# ─────────────────────────────────────────────
# SELESAI
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  🏆  Training pipeline selesai!")
print(f"  Weights  : {HOME}/runs/detect/train/weights/best.pt")
print(f"  Plots    : {HOME}/runs/detect/train/")
print(f"  Preview  : {HOME}/runs/detect/inference_preview.png")
print("=" * 60)
