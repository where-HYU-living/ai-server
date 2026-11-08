"""CubiCasa5K 멀티태스크 네트워크 어댑터: 평면도 이미지 -> 방/아이콘 마스크."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "CubiCasa5k"
WEIGHTS = ROOT / "weights" / "model_best_val_loss_var.pkl"

# 12개 방 클래스, 11개 아이콘 클래스 (CubiCasa5K)
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living", "Bed", "Bath",
                "Entry", "Railing", "Storage", "Garage", "Undefined"]
ICON_CLASSES = ["None", "Window", "Door", "Closet", "Electrical", "Toilet", "Sink",
                "Sauna", "Fireplace", "Bathtub", "Chimney"]
SPLIT = (21, 12, 11)  # heatmaps, rooms, icons


class ModelNotLoaded(RuntimeError):
    pass


@dataclass
class Segmentation:
    rooms: np.ndarray   # HxW int, ROOM_CLASSES 인덱스
    icons: np.ndarray   # HxW int, ICON_CLASSES 인덱스
    image_rgb: np.ndarray
    ratio: tuple[float, float] = (1.0, 1.0)  # 작업 해상도 / 원본 (x, y)


class CubiCasaSegmenter:
    def __init__(self, device: str | None = None, max_side: int = 640):
        if not WEIGHTS.exists() or not VENDOR.exists():
            raise ModelNotLoaded(f"가중치/코드 없음: {WEIGHTS}")
        import torch
        sys.path.insert(0, str(VENDOR))
        from floortrans.models.hg_furukawa_original import hg_furukawa_original

        self.torch = torch
        self.max_side = max_side
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        model = hg_furukawa_original(n_classes=51)  # init_weights 생략: 학습된 가중치로 덮어씀
        n = sum(SPLIT)
        model.conv4_ = torch.nn.Conv2d(256, n, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(n, n, kernel_size=4, stride=4)
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        self.model = model.eval().to(self.device)

    def segment(self, image_rgb: np.ndarray) -> Segmentation:
        torch = self.torch
        import cv2

        h, w = image_rgb.shape[:2]
        scale = self.max_side / max(h, w)
        small = cv2.resize(image_rgb, (max(4, int(w * scale) // 4 * 4), max(4, int(h * scale) // 4 * 4)))
        x = torch.from_numpy(small.astype(np.float32)).permute(2, 0, 1)[None] / 255.0
        x = (x - 0.5) * 2  # [-1, 1]
        with torch.no_grad():
            pred = self.model(x.to(self.device)).cpu()
        sh, sw = small.shape[:2]
        pred = torch.nn.functional.interpolate(pred, size=(sh, sw), mode="bilinear", align_corners=True)
        a, b, c = SPLIT
        rooms = pred[0, a:a + b].argmax(0).numpy().astype(np.uint8)
        icons = pred[0, a + b:a + b + c].argmax(0).numpy().astype(np.uint8)
        return Segmentation(rooms=rooms, icons=icons, image_rgb=small, ratio=(sw / w, sh / h))
