"""SQLite 持久化：患者档案、胸片样本、AI 结果、医生复核操作日志。"""
import sqlite3
import json
from datetime import datetime
from typing import Optional

from app.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            chief_complaint TEXT,
            history TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            heatmap_path TEXT,
            modality TEXT,
            ai_results TEXT,            -- JSON: [{pathology, score, positive, high_alert}]
            review_status TEXT DEFAULT 'pending',  -- pending/confirmed/modified/revoked
            final_labels TEXT,          -- JSON: 医生确认/修改后的标签
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS review_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            action TEXT NOT NULL,       -- confirm/modify/revoke
            operator TEXT,
            detail TEXT,                -- 操作详情(JSON 或文本)
            timestamp TEXT NOT NULL,
            FOREIGN KEY (study_id) REFERENCES studies(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            findings TEXT,
            impression TEXT,
            pdf_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (study_id) REFERENCES studies(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------- 患者 ---------------------------------------
def add_patient(name, gender, age, chief_complaint, history) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO patients (name, gender, age, chief_complaint, history, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (name, gender, age, chief_complaint, history, _now()),
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def update_patient(pid, name, gender, age, chief_complaint, history) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE patients SET name=?, gender=?, age=?, chief_complaint=?, history=? WHERE id=?",
        (name, gender, age, chief_complaint, history, pid),
    )
    conn.commit()
    conn.close()


def list_patients(keyword: Optional[str] = None):
    conn = get_conn()
    if keyword:
        like = f"%{keyword}%"
        rows = conn.execute(
            "SELECT * FROM patients WHERE name LIKE ? OR CAST(id AS TEXT)=? ORDER BY id DESC",
            (like, keyword),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(pid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ----------------------------- 检查/胸片 -----------------------------------
def add_study(patient_id, image_path, modality, ai_results, heatmap_path=None) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO studies (patient_id, image_path, heatmap_path, modality, ai_results, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (patient_id, image_path, heatmap_path, modality, json.dumps(ai_results, ensure_ascii=False), _now()),
    )
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def set_heatmap(study_id, heatmap_path) -> None:
    conn = get_conn()
    conn.execute("UPDATE studies SET heatmap_path=? WHERE id=?", (heatmap_path, study_id))
    conn.commit()
    conn.close()


def list_studies(patient_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM studies WHERE patient_id=? ORDER BY id DESC", (patient_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_study(study_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_review(study_id, status, final_labels) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE studies SET review_status=?, final_labels=? WHERE id=?",
        (status, json.dumps(final_labels, ensure_ascii=False), study_id),
    )
    conn.commit()
    conn.close()


# ----------------------------- 复核日志 -----------------------------------
def add_log(study_id, action, operator, detail) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO review_logs (study_id, action, operator, detail, timestamp)"
        " VALUES (?,?,?,?,?)",
        (study_id, action, operator, json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail, _now()),
    )
    conn.commit()
    conn.close()


def list_logs(study_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM review_logs WHERE study_id=? ORDER BY id DESC", (study_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ----------------------------- 报告 ---------------------------------------
def add_report(study_id, findings, impression, pdf_path) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO reports (study_id, findings, impression, pdf_path, created_at)"
        " VALUES (?,?,?,?,?)",
        (study_id, findings, impression, pdf_path, _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid
