"""智能报告生成：调用云端大模型(OpenAI 兼容接口)生成结构化诊断文本，并导出 PDF。"""
import json
from datetime import datetime
from typing import Dict, List, Optional

import requests

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
    REPORT_DIR,
)

SYSTEM_PROMPT = (
    "你是一名资深的影像科医生助手。请严格依据给定的 AI 结构化筛查指标，"
    "撰写规范的中文胸部 X 线诊断描述。只能基于给定指标进行客观描述，"
    "不得编造未提供的检查结果。输出必须是 JSON，包含两个字段："
    '"findings"（胸部X线表现）与 "impression"（诊断建议）。'
)


def _build_user_prompt(patient: Dict, results: List[Dict]) -> str:
    pos = [r for r in results if r.get("positive")]
    lines = [
        f"患者信息：姓名={patient.get('name','')}, 性别={patient.get('gender','')}, "
        f"年龄={patient.get('age','')}, 主诉={patient.get('chief_complaint','')}, "
        f"既往史={patient.get('history','')}",
        "",
        "AI 结构化筛查结果（疾病: 预测概率）：",
    ]
    for r in results:
        flag = "【高警示】" if r.get("high_alert") else ""
        lines.append(f"- {r['pathology_cn']}({r['pathology']}): {r['score']:.3f} {flag}")
    lines.append("")
    if pos:
        lines.append("阳性发现：" + "、".join(r["pathology_cn"] for r in pos))
    else:
        lines.append("未见明显阳性发现。")
    lines.append("")
    lines.append('请输出 JSON：{"findings": "...", "impression": "..."}')
    return "\n".join(lines)


def generate_report_text(patient: Dict, results: List[Dict]) -> Dict[str, str]:
    """调用云端大模型生成 findings / impression。无 API Key 时回退到本地模板。"""
    if not LLM_API_KEY:
        return _fallback_report(patient, results)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(patient, results)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        return {
            "findings": data.get("findings", ""),
            "impression": data.get("impression", ""),
        }
    except json.JSONDecodeError:
        return {"findings": content, "impression": ""}


def _fallback_report(patient: Dict, results: List[Dict]) -> Dict[str, str]:
    """无网络/无 API Key 时的本地模板报告，保证系统可离线演示。"""
    pos = [r for r in results if r.get("positive")]
    if pos:
        findings = "胸部正位片示：" + "；".join(
            f"{r['pathology_cn']}相关征象（AI 预测概率 {r['score']:.2f}）" for r in pos
        ) + "。其余各肺野未见明显异常密度影。"
        high = [r for r in pos if r.get("high_alert")]
        if high:
            impression = (
                "考虑存在 " + "、".join(r["pathology_cn"] for r in high)
                + " 等高紧急程度异常，建议立即结合临床并由上级医师优先复核，必要时进一步行 CT 检查。"
            )
        else:
            impression = (
                "考虑存在 " + "、".join(r["pathology_cn"] for r in pos)
                + " 可能，建议结合临床随访复查。"
            )
    else:
        findings = "胸部正位片示：双肺纹理清晰，心影大小形态未见异常，膈面光整，肋膈角锐利，未见明显活动性病变。"
        impression = "本次 AI 筛查未见明显异常，建议结合临床。"
    return {"findings": findings, "impression": impression}


# ---------------------------------------------------------------------------
# PDF 导出
# ---------------------------------------------------------------------------
def export_pdf(
    patient: Dict,
    results: List[Dict],
    report_text: Dict[str, str],
    image_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    study_id: Optional[int] = None,
) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # 注册中文字体（优先使用系统自带 CJK 字体）
    font_name = "Helvetica"
    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ):
        try:
            pdfmetrics.registerFont(TTFont("CJK", fp))
            font_name = "CJK"
            break
        except Exception:
            continue

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(REPORT_DIR / f"report_{patient.get('id','x')}_{ts}.pdf")
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    y = height - 25 * mm

    def line(text, size=10, dy=6 * mm, bold=False):
        nonlocal y
        c.setFont(font_name, size)
        c.drawString(20 * mm, y, text)
        y -= dy

    c.setFont(font_name, 16)
    c.drawCentredString(width / 2, y, "胸部 X 线 AI 辅助诊断报告")
    y -= 12 * mm

    line(f"报告编号: {study_id if study_id else '-'}    生成时间: {datetime.now():%Y-%m-%d %H:%M}")
    line(
        f"姓名: {patient.get('name','')}    性别: {patient.get('gender','')}    年龄: {patient.get('age','')}"
    )
    line(f"主诉: {patient.get('chief_complaint','')}")
    line(f"既往史: {patient.get('history','')}")
    y -= 3 * mm

    line("一、AI 筛查结构化指标", size=12)
    for r in results:
        if r.get("positive"):
            tag = "【高警示】" if r.get("high_alert") else ""
            line(f"  · {r['pathology_cn']} ({r['pathology']}): {r['score']:.3f} {tag}", size=10, dy=5 * mm)
    if not any(r.get("positive") for r in results):
        line("  · 未见明显阳性发现", size=10, dy=5 * mm)
    y -= 2 * mm

    # 影像（原图 + 热力图）
    img_w = 70 * mm
    try:
        if image_path:
            c.drawImage(ImageReader(image_path), 20 * mm, y - 55 * mm, width=img_w, height=55 * mm, preserveAspectRatio=True)
        if heatmap_path:
            c.drawImage(ImageReader(heatmap_path), 105 * mm, y - 55 * mm, width=img_w, height=55 * mm, preserveAspectRatio=True)
        y -= 60 * mm
    except Exception:
        pass

    def paragraph(title, body):
        nonlocal y
        line(title, size=12)
        c.setFont(font_name, 10)
        # 简单自动换行
        import textwrap
        for chunk in textwrap.wrap(body, width=46):
            if y < 25 * mm:
                c.showPage(); y = height - 25 * mm; c.setFont(font_name, 10)
            c.drawString(20 * mm, y, chunk)
            y -= 6 * mm
        y -= 3 * mm

    paragraph("二、胸部X线表现", report_text.get("findings", ""))
    paragraph("三、诊断建议", report_text.get("impression", ""))

    c.setFont(font_name, 8)
    c.drawString(20 * mm, 15 * mm, "声明：本报告由 AI 系统辅助生成，仅供医生参考，最终诊断以影像科医师签字为准。")
    c.save()
    return out_path
