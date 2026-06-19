"""病灶可视化：基于 Grad-CAM 生成模型关注区域热力图并叠加到原胸片。"""
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from app.inference import load_model, preprocess


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def grad_cam(img: np.ndarray, pathology: str) -> np.ndarray:
    """对指定病理生成 Grad-CAM 激活图，返回 [0,1] 的 224x224 数组。

    手动复现 DenseNet 前向（取 self.features 的特征图作为目标层），
    避免对含 inplace ReLU 的模块注册 backward hook 导致的报错。
    """
    model = load_model()
    tensor = preprocess(img)

    feats = model.features(tensor)   # (1, C, h, w) -> norm5 输出
    feats.retain_grad()
    pooled = F.adaptive_avg_pool2d(torch.relu(feats), (1, 1)).flatten(1)
    logits = model.classifier(pooled)[0]

    idx = list(model.pathologies).index(pathology)
    model.zero_grad()
    logits[idx].backward()

    acts = feats[0].detach()                     # (C, h, w)
    grads = feats.grad[0]                         # (C, h, w)
    weights = grads.mean(dim=(1, 2))             # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = cam.detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_heatmap(img: np.ndarray, pathology: str, alpha: float = 0.4) -> np.ndarray:
    """生成热力图叠加到原胸片，返回 BGR 彩色图（用于保存/展示）。"""
    cam = grad_cam(img, pathology)
    base = _to_uint8_gray(img)
    base = cv2.resize(base, (224, 224))
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)
    return overlay


def save_overlay(img: np.ndarray, pathology: str, out_path: str) -> Optional[str]:
    overlay = overlay_heatmap(img, pathology)
    cv2.imwrite(out_path, overlay)
    return out_path
