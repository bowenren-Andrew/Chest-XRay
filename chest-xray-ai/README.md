# 胸部 X 线影像智能筛查与辅助诊断系统

面向基层医疗及体检场景的**桌面级胸片 AI 智能筛查与辅助诊断系统**。本地无需 GPU，
直接从开源社区加载他人预训练好的胸片识别模型，跑通"医学影像智能推理 → 人机协同复核 → 诊断报告生成"完整工作流。

## 五大功能

| 功能 | 说明 |
|------|------|
| ① 疾病智能识别 | 加载开源预训练模型 `torchxrayvision DenseNet121 (densenet121-res224-all)`，CPU 实时识别 **14 类肺部异常**；按临床紧急程度分组，自动将 **气胸 / 肺水肿 / 肺实变 / 胸腔积液 / 肿块** 5 类高警示样本**红色高亮预警** |
| ② 病灶可视化 | 基于 **Grad-CAM** 生成热力图，叠加显示模型关注区域；支持 PNG / JPG / **DICOM** 上传、预览与管理 |
| ③ 患者档案管理 | SQLite 持久化患者信息（姓名/性别/年龄/主诉/既往史），档案与胸片、AI 结果**关联绑定**，支持检索与**历史影像对比** |
| ④ 人机协同复核 | 医生可对 AI 结果执行**确认 / 修改 / 撤销**，全程记录**带时间戳的操作日志**，保障责任可追溯 |
| ⑤ 智能报告生成 | 将 AI 结构化指标输入**受控提示词**，调用**云端大模型**（DeepSeek / 通义千问等 OpenAI 兼容接口）生成规范的"胸部X线表现"与"诊断建议"，一键**导出 PDF** |

> 14 类：肺不张、心脏肥大、肺实变、肺水肿、胸腔积液、肺气肿、肺纤维化、疝气、浸润、肿块、结节、胸膜增厚、肺炎、气胸

## 目录结构

```
chest-xray-ai/
├── app/
│   ├── config.py      # 标签/高警示分组/路径/大模型配置
│   ├── db.py          # SQLite：患者、检查、复核日志、报告
│   ├── inference.py   # 预训练模型加载 + 14 类推理 + 分组
│   ├── heatmap.py     # Grad-CAM 热力图
│   ├── report.py      # 云端大模型报告生成 + PDF 导出
│   └── main.py        # Streamlit 工作台（界面）
├── scripts/selftest.py# 端到端自检（无界面）
├── sample_data/       # 示例胸片
├── requirements.txt
└── README.md
```

## 一、环境准备（Python 3.9+）

```bash
# 1. 创建虚拟环境（推荐 Anaconda 或 venv）
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. 安装依赖（CPU 版 torch）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 二、（可选）配置云端大模型

不配置也能运行——系统会自动回退到内置本地模板生成报告。
若要启用真实大模型，设置以下环境变量（以 DeepSeek 为例）：

```bash
export LLM_API_KEY=你的密钥
export LLM_BASE_URL=https://api.deepseek.com/v1   # 通义千问兼容地址: https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=deepseek-chat                      # 通义千问: qwen-plus
```
Windows PowerShell：`$env:LLM_API_KEY="你的密钥"`

## 三、运行与测试

### 方式 A：命令行自检（最快验证模型与全流程）

```bash
# 下载一张示例胸片（首次）
curl -L -o sample_data/sample_cxr.png https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png

python scripts/selftest.py
```
预期输出：依次打印 Top-5 疾病概率、高警示列表，并在 `sample_data/` 生成热力图、在 `reports/` 生成 PDF，最后显示 `✅ 全流程自检通过！`
（首次运行会自动从开源社区下载约 30MB 预训练权重到 `models/`）

### 方式 B：启动可视化工作台

```bash
streamlit run app/main.py
```
浏览器打开 `http://localhost:8501`，操作流程：

1. 进入 **② 患者档案管理 → 新增/编辑**，录入一名患者并保存。
2. 进入 **① 智能筛查/上传**，选择该患者，上传胸片（可用 `sample_data/sample_cxr.png`），点击 **运行 AI 智能识别** → 查看 14 类概率与红色高警示分组。
3. 在 **病灶可视化** 选择病灶 → **生成热力图**，对比原图与热力图。
4. 在 **人机协同复核** 执行 确认/修改/撤销，查看带时间戳的操作日志。
5. 在 **智能报告生成** 点击 **调用大模型生成报告** → 编辑文本 → **导出 PDF**。
6. 进入 **③ 历史影像对比** 查看同一患者多次检查的纵向对比。

## 四、第三方库

`torch / torchvision / torchxrayvision`（本地 CPU 推理）、`opencv-python / Pillow`（图像处理与热力图叠加）、
`pydicom`（DICOM 解析）、`streamlit`（界面）、`requests`（调用云端大模型）、`reportlab`（PDF 导出）。

## 五、模型来源

模型权重通过开源渠道（[torchxrayvision](https://github.com/mlmed/torchxrayvision)）自动获取，
为在 NIH ChestX-ray14、CheXpert、MIMIC-CXR 等多个公开数据集上预训练的 DenseNet121，**本课题不进行模型训练**。

> ⚠️ 免责声明：AI 结果仅作辅助参考，最终诊断须由影像科医师确认。
