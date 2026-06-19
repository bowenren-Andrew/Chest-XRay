"""端到端自检脚本：不启动界面，直接跑通推理→热力图→报告→PDF 全流程。

用法：python scripts/selftest.py [可选的胸片路径]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import inference, db
from app import heatmap as hm
from app import report as rpt


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_cxr.png"
    if not os.path.exists(img_path):
        print(f"[!] 找不到测试图像：{img_path}")
        print("    可先下载示例：curl -L -o sample_data/sample_cxr.png "
              "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png")
        sys.exit(1)

    db.init_db()
    print(f"[1/5] 加载胸片: {img_path}")
    arr = inference.load_image(img_path)

    print("[2/5] AI 推理 14 类疾病 ...")
    results = inference.predict(arr)
    high, normal, neg = inference.split_by_alert(results)
    print("      Top-5 概率:")
    for r in results[:5]:
        flag = " 【高警示】" if r["high_alert"] else ""
        print(f"        - {r['pathology_cn']:6s} ({r['pathology']}): {r['score']:.3f}{flag}")
    print(f"      高警示阳性: {[r['pathology_cn'] for r in high] or '无'}")

    print("[3/5] 生成 Grad-CAM 热力图 ...")
    heat = hm.save_overlay(arr, results[0]["pathology"], "sample_data/selftest_heat.png")
    print(f"      已保存: {heat}")

    print("[4/5] 生成诊断报告文本（无 API Key 时使用本地模板）...")
    patient = {"id": 0, "name": "自检患者", "gender": "男", "age": 50,
               "chief_complaint": "体检", "history": "无"}
    text = rpt.generate_report_text(patient, results)
    print(f"      表现: {text['findings'][:50]}...")

    print("[5/5] 导出 PDF ...")
    pdf = rpt.export_pdf(patient, results, text, image_path=img_path,
                         heatmap_path=heat, study_id=0)
    print(f"      已保存: {pdf}  ({os.path.getsize(pdf)} bytes)")
    print("\n✅ 全流程自检通过！")


if __name__ == "__main__":
    main()
