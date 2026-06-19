"""全局配置：疾病标签、高警示分组、路径与大模型 API 设置。"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_STORE = DATA_DIR / "images"          # 上传的胸片原图存档
HEATMAP_STORE = DATA_DIR / "heatmaps"      # 生成的热力图存档
REPORT_DIR = BASE_DIR / "reports"          # 导出的 PDF 报告
MODEL_DIR = BASE_DIR / "models"            # 预训练模型权重缓存
DB_PATH = DATA_DIR / "chest_xray.db"

for _d in (DATA_DIR, IMAGE_STORE, HEATMAP_STORE, REPORT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# torchxrayvision 权重默认下载到 ~/.torchxrayvision，可重定向到项目内
os.environ.setdefault("TORCHXRAYVISION_CACHE_DIR", str(MODEL_DIR))

# ---------------------------------------------------------------------------
# 14 类肺部异常（ChestX-ray14 标准类别，中英文映射）
# ---------------------------------------------------------------------------
# torchxrayvision DenseNet(weights="densenet121-res224-all") 输出 18 个病理，
# 这里筛选课题要求的 14 类，并给出中文名。
PATHOLOGY_CN = {
    "Atelectasis": "肺不张",
    "Cardiomegaly": "心脏肥大",
    "Consolidation": "肺实变",
    "Edema": "肺水肿",
    "Effusion": "胸腔积液",
    "Emphysema": "肺气肿",
    "Fibrosis": "肺纤维化",
    "Hernia": "疝气",
    "Infiltration": "浸润",
    "Mass": "肿块",
    "Nodule": "结节",
    "Pleural_Thickening": "胸膜增厚",
    "Pneumonia": "肺炎",
    "Pneumothorax": "气胸",
}

# 课题要求展示的 14 类（模型输出名称）
TARGET_PATHOLOGIES = list(PATHOLOGY_CN.keys())

# 高警示（临床高紧急程度）类别：气胸、肺水肿、肺实变、胸腔积液、肿块
HIGH_ALERT = {"Pneumothorax", "Edema", "Consolidation", "Effusion", "Mass"}

# 判定为“阳性”的默认概率阈值
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# 云端大模型（智能报告生成）配置
# 支持任意 OpenAI 兼容接口：DeepSeek / 阿里云通义千问(DashScope 兼容模式) 等。
# 通过环境变量配置，避免把密钥写进代码。
# ---------------------------------------------------------------------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))


def cn(name: str) -> str:
    """英文病理名 -> 中文名。"""
    return PATHOLOGY_CN.get(name, name)
