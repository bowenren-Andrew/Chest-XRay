"""胸部 X 线影像智能筛查与辅助诊断系统 —— Streamlit 工作台。

运行：streamlit run app/main.py
"""
import os
import sys
import json
import shutil
from datetime import datetime

# 允许 `streamlit run app/main.py` 时正确导入 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from app import db
from app.config import (
    IMAGE_STORE,
    HEATMAP_STORE,
    TARGET_PATHOLOGIES,
    HIGH_ALERT,
    DEFAULT_THRESHOLD,
    LLM_API_KEY,
    LLM_MODEL,
    cn,
)
from app import inference
from app import heatmap as hm
from app import report as rpt

st.set_page_config(page_title="胸片AI智能筛查与辅助诊断系统", layout="wide", page_icon="🫁")
db.init_db()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _save_upload(uploaded_file, patient_id) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"p{patient_id}_{ts}{ext}"
    dst = str(IMAGE_STORE / fname)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst


def _png_preview_path(image_path: str) -> str:
    """把任意格式（含 DICOM）转成 PNG 以便 streamlit / PDF 展示。"""
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        return image_path
    import cv2
    arr = inference.load_image(image_path)
    a = arr.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255
    out = image_path + ".png"
    cv2.imwrite(out, a.astype(np.uint8))
    return out


# ---------------------------------------------------------------------------
# 侧边栏导航
# ---------------------------------------------------------------------------
st.sidebar.title("🫁 胸片AI辅助诊断")
page = st.sidebar.radio(
    "功能导航",
    ["① 智能筛查 / 上传", "② 患者档案管理", "③ 历史影像对比", "④ 系统说明"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"大模型: {'已配置 ' + LLM_MODEL if LLM_API_KEY else '未配置(将用本地模板)'}"
)
threshold = st.sidebar.slider("阳性判定阈值", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)


def render_results(results):
    """分组渲染：高警示(红) / 一般阳性(橙) / 概率明细表。"""
    high, normal, negative = inference.split_by_alert(results)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔴 高警示发现（优先复核）")
        if high:
            for r in high:
                st.markdown(
                    f"<div style='background:#ffe5e5;border-left:6px solid #d00;"
                    f"padding:8px;margin:4px 0;border-radius:4px'>"
                    f"<b style='color:#b00'>⚠ {r['pathology_cn']}</b> "
                    f"({r['pathology']}) — 概率 <b>{r['score']:.3f}</b></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("未检出高警示类别（气胸/肺水肿/肺实变/胸腔积液/肿块）")
    with c2:
        st.markdown("#### 🟠 一般阳性发现")
        if normal:
            for r in normal:
                st.markdown(
                    f"<div style='background:#fff3e0;border-left:6px solid #fb8c00;"
                    f"padding:8px;margin:4px 0;border-radius:4px'>"
                    f"<b>{r['pathology_cn']}</b> ({r['pathology']}) — 概率 {r['score']:.3f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("无一般阳性发现")

    st.markdown("#### 📊 14 类疾病概率明细")
    df = pd.DataFrame(
        [
            {
                "疾病": r["pathology_cn"],
                "英文": r["pathology"],
                "概率": r["score"],
                "判定": "阳性" if r["positive"] else "阴性",
                "高警示": "是" if r["high_alert"] else "",
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("疾病")["概率"])


# ===========================================================================
# 页面①：智能筛查 / 上传
# ===========================================================================
if page == "① 智能筛查 / 上传":
    st.header("① 疾病智能识别 · 病灶热力图 · 报告生成")

    patients = db.list_patients()
    if not patients:
        st.warning("暂无患者，请先到『② 患者档案管理』录入患者。")
    else:
        opt = {f"#{p['id']} {p['name']}（{p['gender']},{p['age']}岁）": p["id"] for p in patients}
        sel = st.selectbox("选择关联患者", list(opt.keys()))
        patient_id = opt[sel]

        uploaded = st.file_uploader(
            "上传胸片（支持 PNG / JPG / DICOM）", type=["png", "jpg", "jpeg", "dcm", "dicom"]
        )

        if uploaded is not None:
            img_path = _save_upload(uploaded, patient_id)
            modality = "DICOM" if img_path.lower().endswith((".dcm", ".dicom")) else "IMG"
            preview = _png_preview_path(img_path)
            st.image(preview, caption="胸片预览", width=380)

            if st.button("🔍 运行 AI 智能识别", type="primary"):
                with st.spinner("模型推理中（首次会下载预训练权重，请稍候）..."):
                    arr = inference.load_image(img_path)
                    results = inference.predict(arr, threshold=threshold)
                    sid = db.add_study(patient_id, img_path, modality, results)
                    st.session_state["last_study"] = sid
                    st.session_state["last_arr_path"] = img_path
                st.success(f"识别完成，已生成检查记录 #{sid}")
                render_results(results)

        # 已有结果的检查 -> 热力图 / 复核 / 报告
        sid = st.session_state.get("last_study")
        if sid:
            study = db.get_study(sid)
            results = json.loads(study["ai_results"])
            patient = db.get_patient(patient_id)
            st.markdown("---")

            # ---- 热力图 ----
            st.subheader("② 病灶可视化（Grad-CAM 热力图）")
            pos_labels = [r["pathology"] for r in results if r["positive"]] or [results[0]["pathology"]]
            target = st.selectbox(
                "选择要可视化的病灶",
                pos_labels,
                format_func=cn,
            )
            if st.button("生成热力图"):
                with st.spinner("生成热力图..."):
                    arr = inference.load_image(study["image_path"])
                    out = str(HEATMAP_STORE / f"heat_{sid}_{target}.png")
                    hm.save_overlay(arr, target, out)
                    db.set_heatmap(sid, out)
                c1, c2 = st.columns(2)
                c1.image(_png_preview_path(study["image_path"]), caption="原始胸片", use_container_width=True)
                c2.image(out, caption=f"{cn(target)} 热力图叠加", use_container_width=True)

            # ---- 人机协同复核 ----
            st.markdown("---")
            st.subheader("④ 人机协同复核")
            operator = st.text_input("复核医生", value="影像科医生")
            current_labels = [r["pathology_cn"] for r in results if r["positive"]]
            edited = st.multiselect(
                "确认/修改诊断标签（可增删）",
                options=[cn(p) for p in TARGET_PATHOLOGIES],
                default=current_labels,
            )
            colA, colB, colC = st.columns(3)
            if colA.button("✅ 确认结果"):
                db.update_review(sid, "confirmed", edited)
                db.add_log(sid, "confirm", operator, {"labels": edited})
                st.success("已确认并记录操作日志")
            if colB.button("✏️ 修改诊断"):
                db.update_review(sid, "modified", edited)
                db.add_log(sid, "modify", operator, {"from": current_labels, "to": edited})
                st.success("已保存修改并记录操作日志")
            if colC.button("↩️ 撤销结果"):
                db.update_review(sid, "revoked", [])
                db.add_log(sid, "revoke", operator, {"revoked_from": current_labels})
                st.warning("已撤销该 AI 结果")

            logs = db.list_logs(sid)
            if logs:
                st.markdown("**操作日志（含时间戳）**")
                st.dataframe(
                    pd.DataFrame(
                        [{"时间": l["timestamp"], "操作": l["action"], "医生": l["operator"], "详情": l["detail"]} for l in logs]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # ---- 智能报告生成与导出 ----
            st.markdown("---")
            st.subheader("⑤ 智能报告生成与 PDF 导出")
            if st.button("🧠 调用大模型生成报告"):
                with st.spinner("生成诊断报告中..."):
                    try:
                        text = rpt.generate_report_text(patient, results)
                    except Exception as e:
                        st.error(f"云端大模型调用失败，已回退本地模板：{e}")
                        text = rpt._fallback_report(patient, results)
                    st.session_state["report_text"] = text
            if "report_text" in st.session_state:
                text = st.session_state["report_text"]
                f = st.text_area("胸部X线表现", text.get("findings", ""), height=120)
                im = st.text_area("诊断建议", text.get("impression", ""), height=100)
                if st.button("📄 导出 PDF 报告"):
                    study = db.get_study(sid)
                    pdf = rpt.export_pdf(
                        patient,
                        results,
                        {"findings": f, "impression": im},
                        image_path=_png_preview_path(study["image_path"]),
                        heatmap_path=study.get("heatmap_path"),
                        study_id=sid,
                    )
                    db.add_report(sid, f, im, pdf)
                    with open(pdf, "rb") as fh:
                        st.download_button("⬇️ 下载 PDF", fh, file_name=os.path.basename(pdf), mime="application/pdf")
                    st.success(f"PDF 已生成：{pdf}")


# ===========================================================================
# 页面②：患者档案管理
# ===========================================================================
elif page == "② 患者档案管理":
    st.header("③ 患者档案管理")
    tab1, tab2 = st.tabs(["新增 / 编辑", "查询列表"])

    with tab1:
        edit_id = st.session_state.get("edit_patient")
        p = db.get_patient(edit_id) if edit_id else {}
        with st.form("patient_form"):
            name = st.text_input("姓名*", value=p.get("name", ""))
            gender = st.selectbox("性别", ["男", "女", "其他"], index=["男", "女", "其他"].index(p.get("gender", "男")) if p.get("gender") in ["男","女","其他"] else 0)
            age = st.number_input("年龄", 0, 150, int(p.get("age") or 0))
            cc = st.text_area("主诉", value=p.get("chief_complaint", ""))
            hist = st.text_area("既往病史", value=p.get("history", ""))
            submitted = st.form_submit_button("保存")
            if submitted:
                if not name:
                    st.error("姓名为必填项")
                elif edit_id:
                    db.update_patient(edit_id, name, gender, age, cc, hist)
                    st.session_state.pop("edit_patient", None)
                    st.success("已更新患者信息")
                else:
                    pid = db.add_patient(name, gender, age, cc, hist)
                    st.success(f"已新增患者 #{pid}")

    with tab2:
        kw = st.text_input("按姓名 / ID 检索")
        for p in db.list_patients(kw or None):
            with st.expander(f"#{p['id']} {p['name']}  |  {p['gender']} {p['age']}岁  |  {p['created_at']}"):
                st.write(f"**主诉**：{p['chief_complaint']}")
                st.write(f"**既往史**：{p['history']}")
                studies = db.list_studies(p["id"])
                st.write(f"**关联检查数**：{len(studies)}")
                if st.button("编辑该患者", key=f"edit{p['id']}"):
                    st.session_state["edit_patient"] = p["id"]
                    st.rerun()


# ===========================================================================
# 页面③：历史影像对比
# ===========================================================================
elif page == "③ 历史影像对比":
    st.header("历史影像对比分析")
    patients = db.list_patients()
    if not patients:
        st.info("暂无患者数据")
    else:
        opt = {f"#{p['id']} {p['name']}": p["id"] for p in patients}
        sel = st.selectbox("选择患者", list(opt.keys()))
        studies = db.list_studies(opt[sel])
        if not studies:
            st.info("该患者暂无影像记录")
        else:
            st.write(f"共 {len(studies)} 次检查，按时间排列：")
            cols = st.columns(min(len(studies), 3))
            for i, s in enumerate(studies):
                with cols[i % 3]:
                    st.caption(f"检查#{s['id']} · {s['created_at']} · 状态:{s['review_status']}")
                    try:
                        st.image(_png_preview_path(s["image_path"]), use_container_width=True)
                    except Exception:
                        st.write("(图像缺失)")
                    res = json.loads(s["ai_results"])
                    top = max(res, key=lambda r: r["score"])
                    st.write(f"最高概率：{top['pathology_cn']} {top['score']:.2f}")


# ===========================================================================
# 页面④：系统说明
# ===========================================================================
else:
    st.header("系统说明")
    st.markdown(
        """
本系统为**面向基层医疗及体检场景的桌面级胸片 AI 智能筛查与辅助诊断系统**，覆盖五大功能：

1. **疾病智能识别**：加载开源预训练模型 `torchxrayvision DenseNet121 (densenet121-res224-all)`，
   CPU 实时识别 14 类肺部异常，并将 **气胸/肺水肿/肺实变/胸腔积液/肿块** 5 类高警示样本红色高亮预警。
2. **病灶可视化**：基于 Grad-CAM 生成热力图叠加显示模型关注区域，支持 PNG/JPG/DICOM。
3. **患者档案管理**：SQLite 持久化患者信息，档案与胸片、AI 结果关联绑定，支持检索与历史对比。
4. **人机协同复核**：医生可对 AI 结果执行确认/修改/撤销，全程记录带时间戳的操作日志。
5. **智能报告生成**：调用云端大模型(OpenAI 兼容接口，如 DeepSeek/通义千问)生成规范诊断文本并导出 PDF。

> ⚠️ AI 结果仅作辅助参考，最终诊断需由影像科医师确认。
        """
    )
    st.code(
        "export LLM_API_KEY=你的密钥\n"
        "export LLM_BASE_URL=https://api.deepseek.com/v1   # 或通义千问兼容地址\n"
        "export LLM_MODEL=deepseek-chat",
        language="bash",
    )
