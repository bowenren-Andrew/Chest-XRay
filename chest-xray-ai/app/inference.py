"""胸片 AI 推理：加载开源预训练模型(torchxrayvision)，输出 14 类疾病概率。

模型来源：torchxrayvision 提供的在多个公开 Chest X-ray 数据集(NIH ChestX-ray14、
CheXpert、MIMIC-CXR 等)上预训练的 DenseNet121 (weights="densenet121-res224-all")。
首次运行会自动从开源社区下载权重到本地 models/ 目录，之后纯 CPU 推理。
"""
from functools import lru_cache
from typing import Dict, List

import numpy as np
try:
    import torch
except Exception:
    torch = None

import torchxrayvision as xrv
import skimage.io

from app.config import TARGET_PATHOLOGIES, HIGH_ALERT, DEFAULT_THRESHOLD, cn


@lru_cache(maxsize=1)
def load_model():
    """加载预训练 DenseNet（带缓存，进程内只加载一次）。"""
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def preprocess(img: np.ndarray) -> torch.Tensor:
    """把任意灰度/彩色图像预处理为模型输入张量 (1,1,224,224)。"""
    # 转灰度
    if img.ndim == 3:
        img = img.mean(axis=2)
    # torchxrayvision 要求像素归一化到 [-1024, 1024]
    img = xrv.datasets.normalize(img, maxval=img.max() if img.max() > 0 else 255)
    img = img[None, ...]  # 增加通道维 -> (1, H, W)
    transform = xrv.datasets.XRayCenterCrop()
    img = transform(img)
    img = xrv.datasets.XRayResizer(224)(img)
    tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1,1,224,224)
    return tensor


def load_image(path: str) -> np.ndarray:
    """读取 PNG/JPG/DICOM 胸片为 numpy 数组。"""
    if path.lower().endswith((".dcm", ".dicom")):
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # DICOM 可能是 MONOCHROME1（反相）
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        return arr
    return skimage.io.imread(path).astype(np.float32)


def read_dicom_metadata(path: str) -> Dict[str, str]:
    """从 DICOM 提取患者元数据（可选功能）。"""
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "name": str(getattr(ds, "PatientName", "")),
        "gender": str(getattr(ds, "PatientSex", "")),
        "age": str(getattr(ds, "PatientAge", "")),
        "modality": str(getattr(ds, "Modality", "")),
    }


def predict(img: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """对一张胸片做推理（Cloud安全版）"""

    # 🚨 1. Cloud模式保护
    if torch is None:
        return [
            {
                "pathology": p,
                "pathology_cn": cn(p),
                "score": 0.0,
                "positive": False,
                "high_alert": p in HIGH_ALERT,
            }
            for p in TARGET_PATHOLOGIES
        ]

    # 🚨 2. 尝试加载模型（防止崩溃）
    try:
        model = load_model()
    except Exception:
        return [
            {
                "pathology": p,
                "pathology_cn": cn(p),
                "score": 0.0,
                "positive": False,
                "high_alert": p in HIGH_ALERT,
            }
            for p in TARGET_PATHOLOGIES
        ]

    # 🚨 3. 预处理
    tensor = preprocess(img)

    # 🚨 4. 推理（安全模式）
    try:
        with torch.no_grad():
            out = model(tensor)[0]
        probs = dict(zip(model.pathologies, out.tolist()))
    except Exception:
        return [
            {
                "pathology": p,
                "pathology_cn": cn(p),
                "score": 0.0,
                "positive": False,
                "high_alert": p in HIGH_ALERT,
            }
            for p in TARGET_PATHOLOGIES
        ]

    # 🚨 5. 正常结果处理
    results = []
    for p in TARGET_PATHOLOGIES:
        score = float(probs.get(p, 0.0))
        results.append(
            {
                "pathology": p,
                "pathology_cn": cn(p),
                "score": round(score, 4),
                "positive": score >= threshold,
                "high_alert": p in HIGH_ALERT,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def split_by_alert(results: List[Dict]):
    """按临床紧急程度分组：高警示阳性 / 一般阳性 / 阴性。"""
    high = [r for r in results if r["high_alert"] and r["positive"]]
    normal = [r for r in results if not r["high_alert"] and r["positive"]]
    negative = [r for r in results if not r["positive"]]
    return high, normal, negative
