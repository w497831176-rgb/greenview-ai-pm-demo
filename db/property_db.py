"""
Property SQLite Database
========================

Lightweight SQLite storage for work orders, knowledge docs, and badcases.
Database file lives on a mounted volume so it persists across container restarts.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.handoff_policy import HANDOFF_TRANSITIONS, is_transition_allowed

# Persist DB on the mounted volume so data survives container restarts.
DB_DIR = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data"))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "property_demo.db"


def now_cn(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current Beijing time as a formatted string."""
    cn_tz = timezone(timedelta(hours=8))
    return datetime.now(cn_tz).strftime(fmt)


def now_cn_dt() -> datetime:
    """Return current Beijing time as a datetime object."""
    cn_tz = timezone(timedelta(hours=8))
    return datetime.now(cn_tz)


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and seed demo data if empty."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS work_orders (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            room_id TEXT NOT NULL,
            contact_name TEXT,
            contact_phone TEXT,
            issue_type TEXT,
            issue_desc TEXT,
            urgency TEXT,
            status TEXT,
            appointment_time TEXT,
            created_at TEXT,
            updated_at TEXT,
            assigned_to TEXT,
            completion_note TEXT,
            rating INTEGER
        )
        """
    )

    # V1.4.3: per-session work order drafts pending explicit user confirmation.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS work_order_drafts (
            session_id TEXT PRIMARY KEY,
            room_id TEXT NOT NULL,
            issue_type TEXT,
            issue_desc TEXT,
            urgency TEXT,
            contact_name TEXT,
            contact_phone TEXT,
            appointment_time TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    # Migration bookkeeping: ensures one-time migrations are not re-run.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS migration_meta (
            key TEXT PRIMARY KEY,
            applied_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            category TEXT,
            index_status TEXT DEFAULT 'pending',
            chunk_count INTEGER DEFAULT 0,
            is_indexed INTEGER DEFAULT 1,
            chunk_size INTEGER DEFAULT 512,
            chunk_overlap INTEGER DEFAULT 64,
            split_strategy TEXT DEFAULT 'auto',
            source_type TEXT DEFAULT 'business'
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS badcases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            category TEXT,
            status TEXT,
            created_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            instructions TEXT,
            category TEXT,
            enabled INTEGER DEFAULT 1,
            trigger_condition TEXT,
            skill_metadata TEXT,
            storage_path TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    # V1.5.2: immutable Skill configuration snapshots.  Skills are business
    # capabilities and need an auditable release/rollback history instead of
    # silently overwriting a prompt in place.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id INTEGER NOT NULL,
            version TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            change_summary TEXT,
            created_by TEXT,
            created_at TEXT,
            UNIQUE(skill_id, version)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_servers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            command TEXT,
            args TEXT,
            env TEXT,
            description TEXT,
            enabled INTEGER DEFAULT 1,
            is_builtin INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            input_schema TEXT NOT NULL,
            tool_metadata TEXT,
            UNIQUE(server_id, name),
            FOREIGN KEY (server_id) REFERENCES mcp_servers(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT,
            instructions TEXT,
            category TEXT,
            enabled INTEGER DEFAULT 1,
            model_id TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            skill_id INTEGER NOT NULL,
            UNIQUE(agent_id, skill_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            config TEXT,
            UNIQUE(agent_id, tool_name)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieval_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            top_k INTEGER DEFAULT 5,
            keyword_weight REAL DEFAULT 0.3,
            semantic_weight REAL DEFAULT 0.7,
            rrf_k INTEGER DEFAULT 60,
            enable_rerank INTEGER DEFAULT 0,
            rerank_model TEXT,
            score_threshold REAL DEFAULT 0.0,
            context_threshold REAL DEFAULT 0.2,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    try:
        cursor.execute("ALTER TABLE retrieval_settings ADD COLUMN context_threshold REAL DEFAULT 0.2")
    except Exception:
        pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            provider TEXT,
            api_key TEXT,
            base_url TEXT,
            model_params TEXT,
            is_default INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            description TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS badcase_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badcase_id INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_detail TEXT,
            status_before TEXT,
            status_after TEXT,
            created_by TEXT,
            created_at TEXT,
            FOREIGN KEY (badcase_id) REFERENCES badcases(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badcase_id INTEGER,
            title TEXT,
            content TEXT,
            category TEXT,
            status TEXT DEFAULT 'draft',
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            token_count INTEGER DEFAULT 0,
            round_token_count INTEGER DEFAULT 0,
            token_detail TEXT,
            citations TEXT,
            activated_skills TEXT,
            route_intent TEXT,
            route_reason TEXT,
            current_agent TEXT,
            current_agent_id TEXT,
            tool_calls TEXT,
            created_at TEXT
        )
        """
    )
    # V1.4.3: add current_agent_id column to existing chat_messages.
    try:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN current_agent_id TEXT")
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            handoff_status TEXT DEFAULT 'none',
            handoff_reason TEXT,
            handoff_requested_at TEXT,
            handoff_active_at TEXT,
            handoff_resolved_at TEXT,
            assigned_to TEXT,
            title TEXT,
            last_message_at TEXT,
            last_message_preview TEXT,
            last_agent TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    # V1.5.3: Human copilot is a responsibility workflow, not just a boolean
    # transfer flag.  All additions are non-destructive migrations so existing
    # owner conversations stay available after upgrade.
    for col, dtype in [
        ("handoff_risk_level", "TEXT"),
        ("handoff_reason_code", "TEXT"),
        ("handoff_queue", "TEXT"),
        ("handoff_package_json", "TEXT"),
        ("handoff_summary", "TEXT"),
        ("handoff_outcome", "TEXT"),
        ("handoff_waiting_at", "TEXT"),
        ("handoff_closed_at", "TEXT"),
        ("handoff_cancelled_at", "TEXT"),
        ("handoff_last_actor", "TEXT"),
        ("handoff_last_action_at", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE chat_sessions ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS handoff_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            status_before TEXT,
            status_after TEXT,
            actor TEXT,
            action_detail TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_handoff_actions_session ON handoff_actions(session_id, id)"
    )

    # Migration: add session metadata columns.
    for col, dtype in [
        ("title", "TEXT"),
        ("last_message_at", "TEXT"),
        ("last_message_preview", "TEXT"),
        ("last_agent", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE chat_sessions ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # Migration: persist the full turn total separately from the vertical answer.
    # A turn can contain both a Router and a vertical Agent model call.
    try:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN round_token_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # Migration: add token_detail column to existing chat_messages table.
    try:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN token_detail TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: add citations / activated_skills columns to existing chat_messages table.
    try:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN citations TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN activated_skills TEXT")
    except sqlite3.OperationalError:
        pass
    for col, dtype in [
        ("route_intent", "TEXT"),
        ("route_reason", "TEXT"),
        ("current_agent", "TEXT"),
        ("tool_calls", "TEXT"),
        ("model_id", "TEXT"),
        ("thinking_enabled", "INTEGER"),
        ("model_selection_reason", "TEXT"),
        ("trace_id", "TEXT"),
        ("status", "TEXT"),
        ("latency_ms", "INTEGER"),
        ("error_summary", "TEXT"),
        ("mcp_calls", "TEXT"),
        ("usage_source", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE chat_messages ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # Migration: add RAG columns to existing knowledge_docs table.
    for col, dtype in [
        ("index_status", "TEXT DEFAULT 'pending'"),
        ("chunk_count", "INTEGER DEFAULT 0"),
        ("is_indexed", "INTEGER DEFAULT 1"),
        ("chunk_size", "INTEGER DEFAULT 512"),
        ("chunk_overlap", "INTEGER DEFAULT 64"),
        ("split_strategy", "TEXT DEFAULT 'auto'"),
        ("source_type", "TEXT DEFAULT 'business'"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE knowledge_docs ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # V1.5.1: keep historical acceptance fixtures for audit, but do not let
    # them pollute owner-facing RAG. Content is retained; only retrieval scope
    # is changed once for obviously named test documents.
    cursor.execute("SELECT 1 FROM migration_meta WHERE key = ?", ("v151_isolate_demo_rag_docs",))
    if not cursor.fetchone():
        cursor.execute(
            """UPDATE knowledge_docs
               SET source_type = 'demo_test', is_indexed = 0
               WHERE title LIKE 'DEMO_TEST_%'
                  OR title LIKE 'BROWSER_%'
                  OR title = '测试 badcase'
                  OR title = '测试问题标准答案（验收用例）'"""
        )
        cursor.execute(
            "INSERT INTO migration_meta (key, applied_at) VALUES (?, ?)",
            ("v151_isolate_demo_rag_docs", now_cn()),
        )

    # Migration: add Skill metadata columns to existing skills table.
    for col, dtype in [
        ("trigger_condition", "TEXT"),
        ("skill_metadata", "TEXT"),
        ("storage_path", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE skills ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # Migration: add is_builtin column to existing mcp_servers table.
    try:
        cursor.execute("ALTER TABLE mcp_servers ADD COLUMN is_builtin INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # Migration: add evidence / source fields to existing badcases table.
    for col, dtype in [
        ("evidence", "TEXT"),
        ("source_message_id", "INTEGER"),
        ("session_id", "TEXT"),
        ("root_cause", "TEXT"),
        ("fix_plan", "TEXT"),
        ("verified_by", "TEXT"),
        ("verified_at", "TEXT"),
        ("closed_at", "TEXT"),
        ("rejected_reason", "TEXT"),
        ("updated_at", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE badcases ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # Migration: add model_id column to skills table for per-skill model routing.
    try:
        cursor.execute("ALTER TABLE skills ADD COLUMN model_id TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: add tool_metadata column to mcp_tools table.
    try:
        cursor.execute("ALTER TABLE mcp_tools ADD COLUMN tool_metadata TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: ensure unique index for mcp_tools (server_id, name).
    try:
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_tools_server_name ON mcp_tools(server_id, name)")
    except sqlite3.OperationalError:
        pass

    conn.commit()

    # Seed data only if tables are empty
    cursor.execute("SELECT COUNT(*) FROM work_orders")
    if cursor.fetchone()[0] == 0:
        _seed_work_orders(cursor)
        _seed_knowledge(cursor)
        _seed_badcases(cursor)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM skills")
    if cursor.fetchone()[0] == 0:
        _seed_skills(cursor)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM mcp_servers")
    if cursor.fetchone()[0] == 0:
        _seed_mcp_servers(cursor)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM agents")
    if cursor.fetchone()[0] == 0:
        _seed_agents(cursor)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM retrieval_settings")
    if cursor.fetchone()[0] == 0:
        _seed_retrieval_settings(cursor)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM model_configs")
    if cursor.fetchone()[0] == 0:
        _seed_model_configs(cursor)
        conn.commit()

    # V1.3 observability creates model_prices/budget/trace tables.  It must run
    # before later price migrations so a brand-new database boots cleanly.
    _migrate_v1_3_observability(cursor)
    conn.commit()

    # V1.4.3: enforce official DeepSeek prices without deleting user-added rows.
    _migrate_official_prices_v143(cursor)
    conn.commit()

    # V1.4.3: one-time model config migration only (no longer overwrites user edits).
    _migrate_model_configs_v143(cursor)
    conn.commit()

    # V1.4.3: one-time agent runtime contract migration only.
    _migrate_runtime_contract_v143(cursor)
    conn.commit()

    # V1.4.3: one-time MCP hygiene migration only.
    _migrate_mcp_hygiene_v143(cursor)
    conn.commit()

    # V1.3.3 Badcase operational closure schema.
    _migrate_v1_3_3_badcase_closure(cursor)
    conn.commit()

    # V1.6: Quality operations loop.  This migration only adds explainability
    # and evaluation records; it never rewrites existing business data.
    _migrate_v1_6_quality_governance(cursor)
    conn.commit()

    # V1.8: versioned runtime releases, immutable per-session snapshots,
    # governed tool/action execution and a single evidence ledger.  This is an
    # additive migration only: existing platform configuration and business
    # records are never rewritten during startup.
    _migrate_v1_8_runtime_convergence(cursor)
    conn.commit()

    # V1.5.2: create a non-destructive baseline snapshot for existing Skills.
    # It does not change instructions, bindings or enablement; it only makes
    # current state traceable before the first governed edit.
    _migrate_skill_governance_v152(cursor)
    conn.commit()

    conn.close()


def _seed_work_orders(cursor):
    orders = [
        ("WO-20260710-001", "3-2-1201", "王先生", "13800138001", "水电", "卫生间天花板滴水，怀疑楼上漏水", "紧急", "待派单", "2026-07-11 09:00", "2026-07-10 14:30", "2026-07-10 14:30", None, None, None),
        ("WO-20260710-002", "5-1-802", "李女士", "13800138002", "门窗", "入户门锁无法反锁", "中", "待处理", "2026-07-11 10:00", "2026-07-10 15:00", "2026-07-10 15:00", "张师傅", None, None),
        ("WO-20260710-003", "2-3-1503", "张先生", "13800138003", "公区", "电梯按钮失灵，按了没反应", "高", "处理中", "2026-07-10 16:00", "2026-07-10 13:00", "2026-07-10 15:30", "李师傅", "已更换按钮面板，测试中", None),
        ("WO-20260710-004", "8-2-601", "陈女士", "13800138004", "家户", "客厅吊灯不亮，已自行更换灯泡仍不亮", "低", "已完成", "2026-07-10 10:00", "2026-07-09 16:00", "2026-07-10 11:00", "王师傅", "线路接触不良，已修复", 5),
        ("WO-20260710-005", "1-1-1102", "刘先生", "13800138005", "水电", "厨房下水道反水，污水外溢", "高", "待派单", "2026-07-11 08:00", "2026-07-10 16:00", "2026-07-10 16:00", None, None, None),
        ("WO-20260710-006", "6-3-902", "赵女士", "13800138006", "门窗", "卧室窗户密封条老化，漏风严重", "中", "处理中", "2026-07-11 14:00", "2026-07-10 10:00", "2026-07-10 16:00", "张师傅", "已测量尺寸，待更换密封条", None),
        ("WO-20260710-007", "4-2-1205", "孙先生", "13800138007", "公区", "楼道灯不亮，影响夜间出行", "低", "已完成", "2026-07-10 09:00", "2026-07-09 09:00", "2026-07-09 15:00", "李师傅", "更换声控灯，恢复正常", 4),
        ("WO-20260710-008", "9-1-1501", "周先生", "13800138008", "水电", "空调一开就跳闸，怀疑电路过载", "紧急", "待派单", "2026-07-10 18:00", "2026-07-10 17:00", "2026-07-10 17:00", None, None, None),
        ("WO-20260710-009", "3-1-602", "吴先生", "13800138009", "家户", "次卧墙面起皮，约 30x40cm 面积", "低", "待处理", "2026-07-12 10:00", "2026-07-10 11:00", "2026-07-10 11:00", "王师傅", None, None),
        ("WO-20260710-010", "7-2-1103", "郑女士", "13800138010", "公区", "小区门禁无法刷卡，多位业主反馈", "中", "处理中", "2026-07-10 15:00", "2026-07-10 12:00", "2026-07-10 14:00", "李师傅", "已联系门禁厂商远程排查", None),
    ]
    cursor.executemany(
        """
        INSERT INTO work_orders
        (id, room_id, contact_name, contact_phone, issue_type, issue_desc, urgency, status,
         appointment_time, created_at, updated_at, assigned_to, completion_note, rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        orders,
    )


def _seed_knowledge(cursor):
    docs = [
        (
            '物业维修服务承诺',
            """YIAI物业物业维修服务承诺

第一章 服务宗旨

YIAI物业始终秉持“业主至上、服务第一”的宗旨，以专业化、标准化、人性化的维修服务，为业主创造安全、舒适、便捷的居住环境。我们承诺对业主报修事项高度重视，做到响应及时、处理高效、结果可追溯，确保每一项维修服务都让业主满意。

第二章 响应时效承诺

一、紧急维修：包括燃气泄漏、火灾、触电、电梯困人、严重漏水、水管爆裂、电路起火等可能危及人身财产安全的事项。接到报修后，物业客服中心在 5 分钟内完成工单登记并通知工程人员，工程人员 30 分钟内到场处置，必要时同步联系 119、120、燃气公司等专业救援力量。

二、一般维修：包括灯具不亮、门锁损坏、门窗漏风、墙面起皮、洁具漏水、开关插座故障等日常维修。接到报修后，客服中心在 30 分钟内完成派单，工程人员 2 小时内响应并与业主确认上门时间，原则上 24 小时内上门维修。

三、公共部位维修：包括电梯故障、楼道灯不亮、门禁失灵、消防设施异常、公共管道堵塞等。物业巡查发现或接到报修后，立即设置安全警示，一般故障 24 小时内修复，重大故障 72 小时内修复并公示维修进度。

第三章 维修质量承诺

一、维修人员上门时统一着装、佩戴工牌、携带完整工具和常用配件，入户前穿戴鞋套，维修过程中保持现场整洁。

二、维修完成后，维修人员应当清理作业现场，向业主说明维修内容、使用注意事项，并请业主现场验收签字。

三、一般维修项目实行 30 天质保期，同一故障在质保期内重复发生的，免费再次维修；涉及材料更换的，材料质保期按照国家相关规定及供应商承诺执行。

四、因维修质量问题导致业主财产损失的，由物业承担相应赔偿责任；因业主使用不当或第三方原因导致的损坏，物业协助维修，费用由责任方承担。

第四章 收费透明承诺

一、业主专有部分维修，物业在上门前明确告知收费标准和预计费用，维修完成后提供明细清单，由业主确认后收取费用。

二、公共区域设施设备维修不向业主个人收取费用，所需费用从物业服务费、公共收益或住宅专项维修资金中列支。

三、严禁维修人员私自收费、加价或收受红包，业主有权拒绝任何未公示的收费项目，并可向物业服务中心投诉举报。

第五章 信息沟通承诺

一、业主可通过 AI 助手、物业 24 小时值班电话、微信公众号、物业服务中心前台等多种渠道报修。

二、维修工单生成后，业主可通过上述渠道实时查询工单状态、维修人员信息、预计完成时间。

三、维修完成后 24 小时内，物业客服人员进行回访，了解业主满意度，收集改进意见。

第六章 投诉与复修

一、业主对维修服务不满意的，可在维修完成后 7 日内提出复修申请，物业应当在 24 小时内安排人员上门核查并免费复修。

二、业主对维修收费有异议的，可向物业服务中心申请复核，物业应当在 3 个工作日内出具书面说明。

三、业主对物业维修服务有投诉的，可拨打物业监督电话、向业主委员会反映或通过政府 12345 热线投诉，物业应当在 24 小时内响应并在 3 个工作日内反馈处理结果。

第七章 服务纪律

一、维修人员应当遵守职业道德，文明礼貌，不得推诿、拖延或拒绝合理维修请求。

二、维修人员应当保护业主隐私，不得泄露业主家庭信息、房屋状况等敏感信息。

三、维修人员不得向业主推销商品或服务，不得接受业主宴请、礼品或任何形式的回扣。

第八章 附则

本承诺自发布之日起实施，适用于YIAI物业所服务小区的所有业主。物业将根据实际服务情况和业主反馈，定期修订完善本承诺，确保服务标准持续提升。""",
            '服务承诺',
        ),
        (
            '维修收费标准',
            """YIAI物业维修收费标准（业主专有部分）

第一章 总则

为规范维修收费行为，保障业主知情权，维护业主和物业服务企业的合法权益，根据《物业管理条例》《物业服务收费管理办法》等规定，结合本小区实际情况，制定本维修收费标准。本标准适用于业主专有部分的日常维修、养护及小修服务，公共区域设施设备维修不向业主收取费用。

第二章 收费构成

维修收费一般由以下部分构成：

一、上门费：维修人员上门勘查、诊断的费用，按次收取，已包含基本检测和简单调试。

二、维修费：维修人员实际作业产生的工时费用，根据维修项目复杂程度、所需工时确定。

三、材料费：更换或新增的零配件、材料费用，按照实际使用数量和单价计算。

四、特殊作业费：涉及高空作业、夜间作业、停水停电配合等特殊情况的附加费用。

五、税金：按照国家税务规定开具发票的税费。

第三章 水电类维修收费标准

一、上门费：50 元/次。

二、普通维修：50-200 元/项，包括更换水龙头、角阀、软管、下水器、普通灯具、开关面板、插座等。

三、中等维修：200-400 元/项，包括疏通下水道、更换马桶配件、维修淋浴花洒、更换浴霸、检修电路分支等。

四、复杂维修：400-800 元/项，包括重新布线、更换配电箱空开、查找隐蔽电路故障、维修地暖分水器、更换热水器进出水阀门等。

五、材料费：水龙头 30-200 元/个，角阀 20-80 元/个，软管 15-60 元/根，LED 吸顶灯 50-300 元/个，开关插座 10-80 元/个，空开 30-150 元/个，具体以品牌和规格为准。

第四章 门窗类维修收费标准

一、上门费：30 元/次。

二、普通维修：30-150 元/项，包括调整门窗合页、更换把手、润滑轨道、更换密封条、更换纱窗压条等。

三、中等维修：150-400 元/项，包括更换门锁、更换滑轮、更换玻璃压条、修复窗框变形、更换闭门器等。

四、复杂维修：400-800 元/项，包括整体拆换入户门、更换大面积玻璃、更换断桥铝窗扇、修复防盗门门框等。

五、材料费：普通门锁 80-300 元/把，防盗门锁芯 100-500 元/个，合页 20-80 元/副，密封条 10-30 元/米，滑轮 30-150 元/个，玻璃按面积和厚度计价。

第五章 家户类维修收费标准

一、上门费：30 元/次。

二、灯具维修：普通灯具更换 30-100 元/盏，吊灯安装 80-300 元/盏，大型水晶灯或智能灯具安装按实际协商。

三、墙面维修：小面积修补 50-150 元/处，局部重新粉刷 30-80 元/平方米，铲除重做 80-150 元/平方米。

四、洁具维修：马桶普通维修 50-200 元/项，更换马桶 200-500 元/个（不含马桶），洗手盆维修 50-200 元/项。

五、家具五金维修：柜门铰链更换 20-50 元/个，抽屉轨道更换 50-150 元/副，晾衣架维修 50-200 元/项。

六、维修费按实际工时和材料计算，上门前维修人员应当向业主说明预估费用，业主确认后方可施工。

第六章 免费维修范围

以下情形不收取上门费和维修费，仅收取材料费（如产生）：

一、房屋尚在质量保修期内，且属于开发商保修范围的维修项目。

二、因物业公共管道、公共线路故障导致业主户内受损，经物业确认属实的。

三、物业组织的设施设备统一检修、保养项目。

四、老年业主、残障业主等特殊群体的简单维修，经物业服务中心登记备案后可减免上门费。

第七章 收费程序与发票

一、维修完成后，维修人员填写《维修服务单》，列明上门费、维修费、材料费、合计金额，由业主签字确认。

二、业主可通过现金、微信、支付宝、刷卡等方式缴费，物业应当场或于 3 个工作日内开具正规发票。

三、业主对收费有异议的，可先支付无争议部分，争议部分由物业服务中心在 3 个工作日内复核并书面答复。

第八章 价格调整与公示

一、本收费标准由物业服务企业制定，经业主大会或业主委员会审议通过后实施。

二、收费标准调整前，物业应当提前 30 日在小区显著位置公示，并征求业主意见。

三、本标准未尽事宜，由业主与物业协商确定，协商不成的可依法申请调解或仲裁。""",
            '收费标准',
        ),
        (
            '维修责任划分说明',
            """YIAI物业物业维修责任划分说明

第一章 总则

为明确物业维修过程中业主、物业服务企业及相关方的责任边界，减少维修纠纷，根据《中华人民共和国民法典》《物业管理条例》《住宅专项维修资金管理办法》等法律法规，结合本小区实际情况，制定本说明。

第二章 业主专有部分维修责任

一、业主专有部分是指业主房屋内部及围护结构以内、专属于业主使用的区域，包括户内墙面、地面、顶棚、门窗、水电管线、灯具、洁具、空调室内机、地暖盘管、入户门及门锁等。

二、业主专有部分的日常维护、小修及因业主使用不当导致的损坏，由业主承担维修责任和费用。

三、业主在装修过程中擅自拆改承重墙、破坏防水层、改变房屋使用功能导致的损坏，由业主自行承担修复责任和由此造成的一切损失。

四、业主专有部分的设施设备超过质保期后发生自然老化、损坏的，由业主承担维修或更换费用；仍在质保期内的，由建设单位或销售单位负责维修。

五、业主出租房屋的，应当与承租人约定维修责任；未约定的，按照法律规定和租赁惯例处理，业主对承租人造成的损坏负有连带维修责任。

第三章 建筑物共有部分维修责任

一、建筑物共有部分是指由全体业主共同所有或共同使用的部分，包括建筑物基础、承重结构、外墙、屋顶、楼梯间、走廊、电梯井、管道井、公共门厅、消防设施、公共照明、门禁系统、监控系统等。

二、共有部分的日常维护、小修由物业服务企业负责，费用从物业服务费中列支。

三、共有部分的中修、大修、更新、改造，经业主共同决定后，可使用住宅专项维修资金；未建立维修资金或资金不足的，由相关业主按建筑面积分摊。

四、因物业服务企业日常维护不到位导致共有部分损坏的，由物业服务企业承担维修责任；因不可抗力、第三方侵权或业主共同使用不当导致损坏的，由责任方或相关业主承担。

第四章 公共设施设备维修责任

一、小区公共设施设备包括道路、绿化、停车场、健身设施、儿童游乐设施、垃圾分类设施、快递柜、充电桩、公共厕所、会所设施等。

二、公共设施设备的日常养护、小修由物业服务企业负责，费用从物业服务费或公共收益中列支。

三、公共设施设备的大修、更新、改造，应当编制方案，经业主大会同意后实施，可使用公共收益或住宅专项维修资金。

四、因市政供水、供电、供气、供热、通信等单位管理维护的设施设备发生故障的，由相应单位负责维修，物业协助联系和现场配合。

第五章 相邻关系导致的维修责任

一、因楼上业主专有部分漏水、渗水导致楼下业主房屋受损的，由楼上业主承担维修责任和赔偿责任，物业协助协调、取证和维修。

二、因公共管道堵塞导致多户业主房屋反水、受损的，由相关业主或物业按照责任比例承担维修费用；无法确定具体责任人的，由受益业主共同分摊。

三、因相邻业主装修、改造、使用不当影响他人房屋安全的，由责任方承担维修和赔偿责任，物业有权制止违规行为并报告相关部门。

第六章 质保期内的维修责任

一、新建住宅在质量保修期内出现的质量问题，由建设单位负责维修；业主应当保留购房合同、质保书等相关资料，便于维权。

二、质保期内，业主发现质量问题可先向物业报修，物业应当在 24 小时内转告建设单位或其委托的维修单位。

三、建设单位拖延或拒绝维修的，业主可向住房城乡建设主管部门投诉，也可依法向人民法院提起诉讼。

第七章 责任争议处理

一、维修责任存在争议的，由物业工程人员上门勘查，结合现场情况、维修记录、相关证据作出初步判定。

二、当事人对初步判定有异议的，可共同委托具有资质的第三方鉴定机构进行鉴定，鉴定费用由申请方预付，最终由责任方承担。

三、争议期间，为防止损失扩大，物业可先行组织应急维修，相关费用由最终确定的责任方承担。

第八章 附则

本说明自发布之日起实施，适用于YIAI物业所服务小区。业主、物业服务企业及相关方应当遵守本说明，共同维护小区良好的维修秩序。""",
            '责任划分',
        ),
        (
            '常见维修问题 FAQ',
            """YIAI物业常见维修问题 FAQ

Q1：家里漏水了怎么办？
A：首先关闭户内总水阀，防止损失扩大；然后拍照记录漏水位置和受损情况；如果是楼上漏水导致楼下受损，及时通知物业，物业会协调楼上业主共同处理；最后通过 AI 助手、物业电话或前台提交维修工单，等待工程人员上门。

Q2：电路跳闸怎么处理？
A：先关闭正在使用的大功率电器，然后尝试复位空气开关。如果复位后立即再次跳闸，说明线路可能存在短路、过载或漏电故障，应停止使用相关电器，联系物业电工上门检查，切勿自行拆改配电箱。

Q3：门锁坏了能免费修吗？
A：入户门锁属于业主专有部分，一般需要业主承担上门费和材料费。具体收费标准可参考《维修收费标准》。如果门锁损坏是因质量问题且在质保期内，可联系销售商或开发商免费维修。

Q4：墙面起皮、发霉是什么原因？
A：墙面起皮、发霉常见原因包括：室内潮湿通风不良、外墙渗水、楼上漏水、装修基层处理不当等。建议先排查水源，处理渗漏问题后再进行墙面修复，否则容易反复。

Q5：卫生间地漏返味怎么办？
A：地漏返味通常是因为水封干涸、地漏密封不严或排水管道负压。可定期向地漏补水保持水封，更换防臭地漏芯，或联系物业检查排水管道通气和密封情况。

Q6：客厅吊灯不亮，换了灯泡也不行？
A：可能是灯座接触不良、驱动电源损坏或线路开关故障。建议先检查开关和线路是否有电，若无法自行判断，可报修物业电工上门检测，避免高空作业危险。

Q7：厨房下水道堵塞怎么解决？
A：轻微堵塞可尝试使用皮搋子或管道疏通剂；如果堵塞严重或反复发生，可能是主排水管或存水弯内有油污、异物堆积，应联系物业使用专业工具疏通，不建议自行拆卸公共管道。

Q8：窗户漏风、漏雨如何处理？
A：窗户漏风通常是密封条老化或窗框变形，可更换密封条或调整窗扇；漏雨可能是排水孔堵塞、密封胶开裂或外墙渗水，需要物业工程人员上门查找原因后维修。

Q9：空调一开就跳闸是什么原因？
A：可能是空调功率过大导致线路过载、空调内部短路、插座接触不良或空开容量不足。建议暂时停用空调，联系物业电工或专业空调维修人员上门检查，切勿强行使用。

Q10：家中暖气不热怎么办？
A：首先检查户内暖气阀门是否开启，排气阀是否积气；其次检查滤网是否堵塞；如果整栋楼或小区普遍不热，可能是供热企业运行问题，应联系供热公司；如仅为户内问题，可报修物业协助处理。

Q11：公共区域灯不亮应该找谁？
A：楼道、电梯厅、地下车库等公共区域的照明属于物业维护范围，业主可通过 AI 助手、物业电话或前台报修，物业应当在 24 小时内修复。

Q12：发现电梯异常怎么办？
A：发现电梯运行异响、困人、门无法关闭等异常情况，应立即停止使用，并通过电梯内紧急呼叫按钮或物业 24 小时电话报告。物业将通知电梯维保单位尽快到场处理，必要时启动应急预案。

Q13：报修后多久能上门？
A：紧急维修 30 分钟内到场，一般维修 24 小时内上门，公共部位维修一般 24 小时内修复。具体以《物业维修服务承诺》为准。

Q14：维修完成后如何验收？
A：维修人员完成维修后应当清理现场，向业主说明维修内容和使用注意事项，请业主现场试用并签字确认。如不满意，可在 7 日内申请复修。

Q15：对维修费用有异议怎么办？
A：可在维修前要求维修人员说明预估费用和收费依据，维修后核对明细清单。如有异议，可向物业服务中心申请复核，物业应当在 3 个工作日内书面答复。""",
            'FAQ',
        ),
        (
            '紧急维修处理流程',
            """YIAI物业紧急维修处理流程

第一章 紧急维修范围

紧急维修是指可能对业主人身安全、财产安全或公共安全造成 immediate 威胁的突发故障和事故，主要包括但不限于以下情形：

一、燃气泄漏：闻到燃气异味、燃气报警器报警、燃气管道破损等。

二、火灾：室内或公共区域出现明火、浓烟、电器起火等。

三、触电：人员触电、电线裸露、配电设施冒烟等。

四、严重漏水：水管爆裂、暖气爆管、楼上大量漏水导致楼下严重受损等。

五、电梯困人：业主或乘客被困在电梯轿厢内无法自行脱困。

六、高空坠物风险：外墙装饰物脱落、阳台构件松动、玻璃幕墙破损等。

七、门禁或消防系统全面失效：小区主出入口门禁失灵、消防水泵无法启动、消防报警系统瘫痪等。

第二章 业主应急措施

一、发现紧急情况时，业主应首先确保自身和家人安全，迅速撤离危险区域。

二、燃气泄漏时，严禁开关电器、使用明火、拨打手机，应立即开窗通风，关闭燃气总阀，到室外安全地点拨打燃气公司抢修电话和物业电话。

三、发生火灾时，如火势较小可使用灭火器扑救，火势较大应立即拨打 119 并撤离，切勿乘坐电梯。

四、发生触电时，应先切断电源，用干燥绝缘物使触电者脱离电源，立即拨打 120 急救电话。

五、电梯困人时，应保持冷静，按下轿厢内紧急呼叫按钮或拨打物业电话，切勿强行扒门、撬门或自行攀爬。

第三章 物业接报与响应

一、物业 24 小时值班电话和 AI 助手均受理紧急报修，接到报修后应立即记录报修人、地址、紧急类型、现场情况、联系方式。

二、紧急报修应在 5 分钟内完成工单登记并通知工程人员、秩序维护人员和项目经理。

三、工程人员应在 30 分钟内携带必要工具和应急物资到场，秩序维护人员应同步赶赴现场维护秩序、疏散人群、设置警戒区域。

四、涉及燃气、消防、电梯等专业救援的，物业应立即拨打 119、120、燃气公司、电梯维保单位电话，并安排人员到出入口引导救援车辆。

第四章 现场处置流程

一、工程人员到场后应迅速评估险情，采取切断水、电、气等紧急措施，防止事故扩大。

二、燃气泄漏处置：关闭总阀、开窗通风、疏散人员、禁止明火和电器操作，等待燃气公司专业人员到场检测修复。

三、火灾处置：组织初期扑救，启动消防泵、排烟风机，引导人员疏散，配合消防队灭火。

四、触电处置：切断电源后，对伤者进行心肺复苏等急救，等待 120 到场。

五、严重漏水处置：关闭相应阀门，使用抽水泵排水，转移受损物品，评估受损范围，协调责任方维修。

六、电梯困人处置：安抚被困人员，使用专业钥匙和工具解救，解救后检查电梯故障原因，确认安全后方可恢复运行。

第五章 后续跟进

一、紧急处置完成后，工程人员应在 2 小时内填写《紧急维修记录表》，详细记录事件经过、处置措施、使用材料、现场照片等。

二、物业应在 24 小时内联系受损业主，了解损失情况，协助业主进行保险理赔或责任追偿。

三、需要后续维修的，应在 24 小时内制定维修方案，明确责任方、费用来源和完成时限，并告知相关业主。

四、紧急维修完成后 48 小时内，物业应进行回访，确认业主满意度，收集改进建议。

第六章 预防与演练

一、物业应定期对供水、供电、供气、消防、电梯等设施设备进行巡检和维护，每月至少一次全面检查，发现隐患及时整改。

二、物业应建立应急物资储备，包括发电机、抽水泵、应急照明、灭火器、沙袋、警示标识等，定期检查更新。

三、物业应每年至少组织一次消防应急演练和一次电梯困人救援演练，提高员工应急处置能力。

四、物业应通过公告栏、微信公众号、业主群等渠道向业主宣传安全常识和紧急联系方式，提高业主自救互救能力。

第七章 责任与追责

一、因物业日常维护不到位、应急响应不及时导致事故扩大的，由物业服务企业承担相应责任。

二、因业主使用不当、违规装修、私改管线等原因导致紧急事故的，由业主承担相应责任和损失。

三、因第三方施工单位、设备维保单位、市政供应单位等原因导致事故的，由责任方承担相应责任，物业协助业主维权。

第八章 附则

本流程自发布之日起实施，适用于YIAI物业所服务小区。物业应根据实际情况和演练反馈，不断完善应急预案，确保紧急维修工作快速、有序、有效。""",
            '紧急流程',
        ),
    ]
    cursor.executemany(
        "INSERT INTO knowledge_docs (title, content, category) VALUES (?, ?, ?)",
        docs,
    )


def _seed_badcases(cursor):
    cases = [
        (
            "AI 错误回答「漏水维修免费」",
            "业主询问'楼上漏水你们负责修吗'，AI 回答'漏水属于物业责任，免费维修'。实际应根据责任划分判定，专有部分漏水一般由业主承担费用。",
            "RAG 幻觉",
            "已修复",
            "2026-07-09 10:00",
        ),
        (
            "AI 未识别'空调跳闸'为紧急问题",
            "业主反馈'空调一开就跳闸'，AI 仅按一般工单处理，未识别电路安全隐患。",
            "意图识别错误",
            "修复中",
            "2026-07-09 14:00",
        ),
        (
            "AI 把'窗户漏风'错误分类为'水电'",
            "业主描述'卧室窗户漏风'，AI 创建的工单 issue_type 被填为'水电'，导致派单错误。",
            "分类错误",
            "待处理",
            "2026-07-10 09:00",
        ),
    ]
    cursor.executemany(
        "INSERT INTO badcases (title, description, category, status, created_at) VALUES (?, ?, ?, ?, ?)",
        cases,
    )


def _seed_skills(cursor):
    now = now_cn("%Y-%m-%d %H:%M")
    skills = [
        (
            "维修工单处理",
            "帮助业主创建维修工单、查询进度、解答维修相关问题。",
            "你是物业维修助手，帮助业主报修、查询工单、解答维修相关问题。",
            "业务技能",
            1,
            "用户要报修、查询工单、创建工单、维修进度",
            now,
            now,
        ),
        (
            "知识库问答",
            "基于物业维修知识库回答收费标准、责任划分、服务承诺等问题。",
            "回答收费标准、维修责任、服务承诺时，必须基于知识库原文；知识库未命中时明确说'需要人工确认'。",
            "业务技能",
            1,
            "用户询问物业服务、收费标准、维修责任、小区规定等知识性问题",
            now,
            now,
        ),
        (
            "孩子托管服务",
            "当业主询问孩子托管服务时激活",
            "当业主询问孩子托管服务时，表达可以全天候8点到晚上21点托管，每天都可，服务项目为免费项目，具体请咨询管理处电话077512345678",
            "业务技能",
            1,
            "用户提到孩子托管、儿童托管、托管服务",
            now,
            now,
        ),
        (
            "宠物托管",
            "当业主询问宠物托管时激活",
            "当业主询问宠物托管服务时，表达可以全天候24小时托管，每天都可，服务项目为100元/天，具体请咨询管理处电话077512345678",
            "业务技能",
            1,
            "用户提到宠物托管、宠物寄养、宠物服务",
            now,
            now,
        ),
    ]
    cursor.executemany(
        "INSERT INTO skills (name, description, instructions, category, enabled, trigger_condition, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        skills,
    )



def _seed_mcp_servers(cursor):
    now = now_cn("%Y-%m-%d %H:%M")
    servers = [
        (
            "weather-server",
            "python",
            json.dumps(["/app/tools/weather_mcp_server.py"]),
            None,
            "天气查询 MCP Server，提供实时天气与天气建议工具。",
            1,
            1,
            now,
            now,
        ),
        (
            "calendar-server",
            "python",
            json.dumps(["/app/tools/calendar_mcp_server.py"]),
            None,
            "日历 MCP Server，提供当前日期与预约时间相关工具。",
            1,
            1,
            now,
            now,
        ),
        (
            "workorder-server",
            "python",
            json.dumps(["/app/tools/db_query_mcp_server.py"]),
            None,
            "工单查询 MCP Server，提供维修工单数量与待办列表只读查询。",
            1,
            1,
            now,
            now,
        ),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO mcp_servers (name, command, args, env, description, enabled, is_builtin, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        servers,
    )


def _seed_agents(cursor):
    now = now_cn("%Y-%m-%d %H:%M")
    agents = [
        (
            "router",
            "路由 Agent",
            "负责识别业主意图并分发给合适的垂直 Agent。",
            "你是一个意图分类专家。根据用户问题，从以下类别中选择最相关的一个：maintenance（维修/工单）、billing（费用/缴费）、complaint（投诉/纠纷）、customer_service（一般客服/咨询）、other（其他/无法判断）。只输出一个分类标签和一句话理由。",
            "orchestration",
            1,
            None,
            now,
            now,
        ),
        (
            "maintenance",
            "维修 Agent",
            "处理维修报修、工单创建与查询。",
            "你是物业维修助手，帮助业主报修、查询工单、解答维修相关问题。",
            "vertical",
            1,
            None,
            now,
            now,
        ),
        (
            "billing",
            "费用 Agent",
            "处理缴费、收费标准、费用争议咨询。",
            "你是物业费用助手，负责解答收费标准、缴费方式、费用争议。涉及费用争议时建议转人工。",
            "vertical",
            1,
            None,
            now,
            now,
        ),
        (
            "complaint",
            "投诉 Agent",
            "处理业主投诉、邻里纠纷、责任争议。",
            "你是物业投诉处理助手，负责安抚业主情绪、记录投诉要点、协调人工跟进。不要自行判定责任。",
            "vertical",
            1,
            None,
            now,
            now,
        ),
        (
            "customer_service",
            "客服 Agent",
            "处理一般咨询、小区规定、服务承诺。",
            "你是物业客服助手，负责解答小区服务、联系方式、一般规定等咨询。",
            "vertical",
            1,
            None,
            now,
            now,
        ),
    ]
    cursor.executemany(
        """
        INSERT INTO agents
        (agent_id, name, description, instructions, category, enabled, model_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        agents,
    )


def _seed_retrieval_settings(cursor):
    now = now_cn("%Y-%m-%d %H:%M")
    cursor.execute(
        """
        INSERT INTO retrieval_settings
        (name, top_k, keyword_weight, semantic_weight, rrf_k, enable_rerank, rerank_model, score_threshold, context_threshold, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("default", 5, 0.3, 0.7, 60, 0, None, 0.0, 0.2, now, now),
    )


def _seed_model_configs(cursor):
    now = now_cn("%Y-%m-%d %H:%M")
    configs = [
        (
            "deepseek-v4-flash",
            "DeepSeek V4 Flash",
            "deepseek",
            None,
            "https://api.deepseek.com",
            json.dumps({"use_thinking": True}),
            1,
            1,
            "常规文本 Router 与垂直 Agent 主力模型",
            now,
            now,
        ),
        (
            "deepseek-v4-pro",
            "DeepSeek V4 Pro",
            "deepseek",
            None,
            "https://api.deepseek.com",
            json.dumps({"use_thinking": True}),
            0,
            1,
            "后台 A/B 与 Darwin 深度复盘模型",
            now,
            now,
        ),
    ]
    cursor.executemany(
        """
        INSERT INTO model_configs
        (model_id, name, provider, api_key, base_url, model_params, is_default, enabled, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        configs,
    )


def _seed_model_prices(cursor):
    """Seed official DeepSeek prices (CNY per 1M tokens) as of 2026-07-17."""
    now = now_cn("%Y-%m-%d %H:%M")
    prices = [
        (
            "deepseek-v4-flash",
            "CNY",
            "2026-07-17",
            1.0,
            0.02,
            2.0,
            0.0,
            "DeepSeek 官方价格表（2026-07-17 校准）",
            1,
            now,
            now,
        ),
        (
            "deepseek-v4-pro",
            "CNY",
            "2026-07-17",
            3.0,
            0.025,
            6.0,
            0.0,
            "DeepSeek 官方价格表（2026-07-17 校准）",
            1,
            now,
            now,
        ),
    ]
    cursor.executemany(
        """
        INSERT INTO model_prices
        (model_id, currency, effective_date, input_price_per_1m, cached_input_price_per_1m,
         output_price_per_1m, reasoning_price_per_1m, source_note, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        prices,
    )


def _migrate_official_prices_v143(cursor):
    """One-time migration to enforce official DeepSeek V4 prices.

    - Disables stale demo/gateway price rows for deepseek-v4-flash/pro.
    - Inserts the official price row if no row with the exact same price
      snapshot already exists.
    - Leaves user-created price rows for other models untouched.
    """
    now = now_cn("%Y-%m-%d %H:%M")
    if _migration_applied(cursor, "v143_official_prices"):
        return

    official = {
        "deepseek-v4-flash": {
            "input": 1.0,
            "cached_input": 0.02,
            "output": 2.0,
            "reasoning": 0.0,
        },
        "deepseek-v4-pro": {
            "input": 3.0,
            "cached_input": 0.025,
            "output": 6.0,
            "reasoning": 0.0,
        },
    }

    for model_id, price in official.items():
        # Disable stale rows that do not match the official snapshot.
        cursor.execute(
            """
            UPDATE model_prices
            SET enabled = 0, updated_at = ?
            WHERE model_id = ?
              AND enabled = 1
              AND (
                COALESCE(input_price_per_1m, -1) != ?
                OR COALESCE(cached_input_price_per_1m, -1) != ?
                OR COALESCE(output_price_per_1m, -1) != ?
                OR COALESCE(reasoning_price_per_1m, -1) != ?
              )
            """,
            (now, model_id, price["input"], price["cached_input"], price["output"], price["reasoning"]),
        )

        # Insert official row only if an exact match is not already present.
        cursor.execute(
            """
            SELECT 1 FROM model_prices
            WHERE model_id = ?
              AND input_price_per_1m = ?
              AND cached_input_price_per_1m = ?
              AND output_price_per_1m = ?
              AND reasoning_price_per_1m = ?
            LIMIT 1
            """,
            (model_id, price["input"], price["cached_input"], price["output"], price["reasoning"]),
        )
        if cursor.fetchone() is None:
            cursor.execute(
                """
                INSERT INTO model_prices
                (model_id, currency, effective_date, input_price_per_1m, cached_input_price_per_1m,
                 output_price_per_1m, reasoning_price_per_1m, source_note, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    "CNY",
                    "2026-07-17",
                    price["input"],
                    price["cached_input"],
                    price["output"],
                    price["reasoning"],
                    "DeepSeek 官方价格表（2026-07-17 校准）",
                    1,
                    now,
                    now,
                ),
            )

    _mark_migration_applied(cursor, "v143_official_prices", now)


def _migrate_model_configs_v143(cursor):
    """One-time migration for model catalog. Existing rows are left untouched."""
    now = now_cn("%Y-%m-%d %H:%M")
    if _migration_applied(cursor, "v143_model_configs"):
        return
    # Only INSERT missing rows; never UPDATE user-edited configurations.
    defaults = [
        ("deepseek-v4-flash", "DeepSeek V4 Flash", json.dumps({"use_thinking": True}), 1, 1, "常规文本 Router 与垂直 Agent 主力模型"),
        ("deepseek-v4-pro", "DeepSeek V4 Pro", json.dumps({"use_thinking": True}), 0, 1, "后台 A/B 与 Darwin 深度复盘模型"),
    ]
    for model_id, name, params, is_default, enabled, description in defaults:
        cursor.execute(
            """
            INSERT OR IGNORE INTO model_configs
            (model_id, name, provider, api_key, base_url, model_params, is_default, enabled, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (model_id, name, "deepseek", None, "https://api.deepseek.com", params, is_default, enabled, description, now, now),
        )
    # If exactly one default exists, leave it; otherwise enforce Flash as the sole default only on first migration.
    cursor.execute("SELECT COUNT(*) FROM model_configs WHERE is_default = 1")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "UPDATE model_configs SET is_default = 1 WHERE model_id = ?",
            ("deepseek-v4-flash",),
        )
    _mark_migration_applied(cursor, "v143_model_configs", now)


def _migration_applied(cursor, key: str) -> bool:
    cursor.execute("SELECT 1 FROM migration_meta WHERE key = ?", (key,))
    return cursor.fetchone() is not None


def _mark_migration_applied(cursor, key: str, applied_at: Optional[str] = None):
    if applied_at is None:
        applied_at = now_cn("%Y-%m-%d %H:%M")
    cursor.execute(
        "INSERT OR REPLACE INTO migration_meta (key, applied_at) VALUES (?, ?)",
        (key, applied_at),
    )


def _migrate_model_configs_legacy(cursor):
    """DEPRECATED: one-time old behavior preserved only for external callers.

    init_db now uses _migrate_model_configs_v143 which never overwrites user edits.
    """
    now = now_cn("%Y-%m-%d %H:%M")
    updates = [
        (
            "deepseek-v4-flash",
            "DeepSeek V4 Flash",
            json.dumps({"use_thinking": True}),
            1,
            1,
            "常规文本 Router 与垂直 Agent 主力模型",
        ),
        (
            "deepseek-v4-pro",
            "DeepSeek V4 Pro",
            json.dumps({"use_thinking": True}),
            0,
            1,
            "后台 A/B 与 Darwin 深度复盘模型",
        ),
    ]
    for model_id, name, params, is_default, enabled, description in updates:
        cursor.execute(
            """
            UPDATE model_configs
            SET name = ?, model_params = ?, is_default = ?, enabled = ?, description = ?, updated_at = ?
            WHERE model_id = ?
            """,
            (name, params, is_default, enabled, description, now, model_id),
        )
    # Ensure no other row accidentally remains default.
    cursor.execute(
        """
        UPDATE model_configs
        SET is_default = 0
        WHERE model_id != ? AND is_default = 1
        """,
        ("deepseek-v4-flash",),
    )


def _migrate_runtime_contract_v143(cursor):
    """One-time runtime contract migration for V1.4.3.

    - Adds is_router column to agents if missing.
    - Inserts missing canonical agents without overwriting existing rows.
    - Does NOT update name/description/instructions/category/enabled of existing agents.
    - Does NOT delete user-created agents or perform rebranding.
    """
    now = now_cn("%Y-%m-%d %H:%M")
    if _migration_applied(cursor, "v143_runtime_contract"):
        return

    # Schema: add is_router column if missing.
    cursor.execute("PRAGMA table_info(agents)")
    columns = {row[1] for row in cursor.fetchall()}
    if "is_router" not in columns:
        cursor.execute("ALTER TABLE agents ADD COLUMN is_router INTEGER DEFAULT 0")
        cursor.execute("UPDATE agents SET is_router = 1 WHERE category IN ('router', 'orchestration')")
        cursor.execute("UPDATE agents SET is_router = 0 WHERE category NOT IN ('router', 'orchestration') OR category IS NULL")

    # Insert missing canonical agents only; never UPDATE existing rows.
    canonical = [
        ("router", "router", "路由 Agent", "负责识别业主意图并路由到对应垂直 Agent。", "你是物业智能客服路由助手，负责判断业主消息属于维修、费用、投诉还是一般客服咨询，并简洁输出意图分类。"),
        ("maintenance", "vertical", "维修 Agent", "处理报修、维修进度、上门预约。", "你是物业维修助手，负责记录业主报修内容、判断紧急程度、创建维修工单。"),
        ("billing", "vertical", "费用 Agent", "处理物业费、停车费、缴费查询。", "你是物业费用助手，负责解释收费标准、查询账单、引导缴费流程。"),
        ("complaint", "vertical", "投诉 Agent", "处理业主投诉、邻里纠纷、责任争议。", "你是物业投诉处理助手，负责安抚业主情绪、记录投诉要点、协调人工跟进。不要自行判定责任。"),
        ("customer_service", "vertical", "客服 Agent", "处理一般咨询、小区规定、服务承诺。", "你是物业客服助手，负责解答小区服务、联系方式、一般规定等咨询。"),
    ]
    for agent_id, category, name, description, instructions in canonical:
        cursor.execute(
            """
            INSERT OR IGNORE INTO agents
            (agent_id, name, description, instructions, category, is_router, enabled, model_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (agent_id, name, description, instructions, category, 1 if category == "router" else 0, 1, None, now, now),
        )

    _mark_migration_applied(cursor, "v143_runtime_contract", now)


def _migrate_runtime_contract(cursor):
    """DEPRECATED: old runtime contract migration preserved for external callers.

    init_db now uses _migrate_runtime_contract_v143 which never overwrites user edits.
    """
    now = now_cn("%Y-%m-%d %H:%M")

    # 1. Add is_router column if missing.
    cursor.execute("PRAGMA table_info(agents)")
    columns = {row[1] for row in cursor.fetchall()}
    if "is_router" not in columns:
        cursor.execute("ALTER TABLE agents ADD COLUMN is_router INTEGER DEFAULT 0")
        cursor.execute("UPDATE agents SET is_router = 1 WHERE category IN ('router', 'orchestration')")
        cursor.execute("UPDATE agents SET is_router = 0 WHERE category NOT IN ('router', 'orchestration') OR category IS NULL")

    # 2. Define canonical agents and ensure each exists with the lowest id possible.
    canonical = [
        ("router", "router", "路由 Agent", "负责识别业主意图并路由到对应垂直 Agent。", "你是物业智能客服路由助手，负责判断业主消息属于维修、费用、投诉还是一般客服咨询，并简洁输出意图分类。"),
        ("maintenance", "vertical", "维修 Agent", "处理报修、维修进度、上门预约。", "你是物业维修助手，负责记录业主报修内容、判断紧急程度、创建维修工单。"),
        ("billing", "vertical", "费用 Agent", "处理物业费、停车费、缴费查询。", "你是物业费用助手，负责解释收费标准、查询账单、引导缴费流程。"),
        ("complaint", "vertical", "投诉 Agent", "处理业主投诉、邻里纠纷、责任争议。", "你是物业投诉处理助手，负责安抚业主情绪、记录投诉要点、协调人工跟进。不要自行判定责任。"),
        ("customer_service", "vertical", "客服 Agent", "处理一般咨询、小区规定、服务承诺。", "你是物业客服助手，负责解答小区服务、联系方式、一般规定等咨询。"),
    ]

    canonical_agent_ids = {}
    for agent_id, category, name, description, instructions in canonical:
        cursor.execute("SELECT id FROM agents WHERE agent_id = ? ORDER BY id LIMIT 1", (agent_id,))
        row = cursor.fetchone()
        if row:
            canonical_row_id = row[0]
            cursor.execute(
                """
                UPDATE agents
                SET agent_id = ?, name = ?, description = ?, instructions = ?, category = ?, is_router = ?, enabled = 1, updated_at = ?
                WHERE id = ?
                """,
                (agent_id, name, description, instructions, category, 1 if category == "router" else 0, now, canonical_row_id),
            )
            # Remove duplicates for this agent_id, keeping canonical_row_id.
            cursor.execute(
                "DELETE FROM agents WHERE agent_id = ? AND id != ?",
                (agent_id, canonical_row_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO agents
                (agent_id, name, description, instructions, category, is_router, enabled, model_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (agent_id, name, description, instructions, category, 1 if category == "router" else 0, 1, None, now, now),
            )
            canonical_row_id = cursor.lastrowid

        canonical_agent_ids[name] = agent_id

        # Migrate skill/tool bindings from rows that share this canonical name but have a different agent_id.
        cursor.execute(
            "SELECT agent_id FROM agents WHERE name = ? AND agent_id != ?",
            (name, agent_id),
        )
        for dup_row in cursor.fetchall():
            dup_agent_id = dup_row[0]
            cursor.execute(
                "INSERT OR IGNORE INTO agent_skills (agent_id, skill_id) SELECT ?, skill_id FROM agent_skills WHERE agent_id = ?",
                (agent_id, dup_agent_id),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO agent_tools (agent_id, tool_name, config) SELECT ?, tool_name, config FROM agent_tools WHERE agent_id = ?",
                (agent_id, dup_agent_id),
            )
            cursor.execute("DELETE FROM agent_skills WHERE agent_id = ?", (dup_agent_id,))
            cursor.execute("DELETE FROM agent_tools WHERE agent_id = ?", (dup_agent_id,))
            cursor.execute("DELETE FROM agents WHERE agent_id = ?", (dup_agent_id,))

    # 3. Remove temp / test agents by name heuristic and migrate bindings first.
    cursor.execute(
        "SELECT agent_id FROM agents WHERE LOWER(name) LIKE '%test%' OR LOWER(name) LIKE '%temp%' OR LOWER(agent_id) LIKE '%test%' OR LOWER(agent_id) LIKE '%temp%'"
    )
    temp_agent_ids = [row[0] for row in cursor.fetchall()]
    fallback_agent_id = canonical_agent_ids.get("维修 Agent") or next(iter(canonical_agent_ids.values()), None)
    for temp_agent_id in temp_agent_ids:
        if fallback_agent_id:
            cursor.execute(
                "INSERT OR IGNORE INTO agent_skills (agent_id, skill_id) SELECT ?, skill_id FROM agent_skills WHERE agent_id = ?",
                (fallback_agent_id, temp_agent_id),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO agent_tools (agent_id, tool_name, config) SELECT ?, tool_name, config FROM agent_tools WHERE agent_id = ?",
                (fallback_agent_id, temp_agent_id),
            )
        cursor.execute("DELETE FROM agent_skills WHERE agent_id = ?", (temp_agent_id,))
        cursor.execute("DELETE FROM agent_tools WHERE agent_id = ?", (temp_agent_id,))
        cursor.execute("DELETE FROM agents WHERE agent_id = ?", (temp_agent_id,))

    # 5. Remove legacy *_agent duplicates created by the old app/main.py seed.
    for agent_id, _, name, _, _ in canonical:
        legacy_id = f"{agent_id}_agent"
        cursor.execute("SELECT agent_id FROM agents WHERE agent_id = ?", (legacy_id,))
        if cursor.fetchone():
            canonical_agent_id = canonical_agent_ids.get(name)
            if canonical_agent_id:
                cursor.execute(
                    "INSERT OR IGNORE INTO agent_skills (agent_id, skill_id) SELECT ?, skill_id FROM agent_skills WHERE agent_id = ?",
                    (canonical_agent_id, legacy_id),
                )
                cursor.execute(
                    "INSERT OR IGNORE INTO agent_tools (agent_id, tool_name, config) SELECT ?, tool_name, config FROM agent_tools WHERE agent_id = ?",
                    (canonical_agent_id, legacy_id),
                )
            cursor.execute("DELETE FROM agent_skills WHERE agent_id = ?", (legacy_id,))
            cursor.execute("DELETE FROM agent_tools WHERE agent_id = ?", (legacy_id,))
            cursor.execute("DELETE FROM agents WHERE agent_id = ?", (legacy_id,))

    # 6. Rebrand user-facing runtime text.
    rebrand_fields = [
        ("knowledge_docs", "title"),
        ("knowledge_docs", "content"),
        ("agents", "name"),
        ("agents", "description"),
        ("agents", "instructions"),
        ("skills", "name"),
        ("skills", "description"),
        ("skills", "instructions"),
        ("skills", "trigger_condition"),
        ("mcp_servers", "name"),
        ("mcp_servers", "description"),
        ("chat_messages", "content"),
    ]
    for table, column in rebrand_fields:
        try:
            cursor.execute(
                f"""
                UPDATE {table}
                SET {column} = REPLACE({column}, '绿景智服', 'YIAI物业')
                WHERE {column} LIKE '%绿景智服%'
                """
            )
            cursor.execute(
                f"""
                UPDATE {table}
                SET {column} = REPLACE({column}, '绿景', 'YIAI物业')
                WHERE {column} LIKE '%绿景%' AND {column} NOT LIKE '%YIAI物业%'
                """
            )
        except Exception:
            pass


def _migrate_mcp_hygiene_v143(cursor):
    """One-time MCP hygiene migration for V1.4.3.

    - Ensures canonical demo MCP server rows exist via an explicit name lookup.
    - Does NOT update command/args/description/enabled of existing servers.
    - Does NOT delete user-created MCP servers or bindings.
    - Does NOT force agent-tool bindings on maintenance/customer_service.
    """
    import json

    now = now_cn("%Y-%m-%d %H:%M")
    if _migration_applied(cursor, "v143_mcp_hygiene"):
        return

    canonical_servers = [
        {
            "name": "weather-server",
            "command": "python",
            "args": ["/app/tools/weather_mcp_server.py"],
            "description": "天气查询 MCP Server，提供实时天气与天气建议工具。",
        },
        {
            "name": "workorder-server",
            "command": "python",
            "args": ["/app/tools/db_query_mcp_server.py"],
            "description": "工单查询 MCP Server，提供维修工单数量与待办列表只读查询。",
        },
        {
            "name": "calendar-server",
            "command": "python",
            "args": ["/app/tools/calendar_mcp_server.py"],
            "description": "日历 MCP Server，提供当前日期与预约时间相关工具。",
        },
    ]

    for server in canonical_servers:
        cursor.execute(
            "SELECT 1 FROM mcp_servers WHERE name = ? LIMIT 1",
            (server["name"],),
        )
        if cursor.fetchone() is None:
            cursor.execute(
                """
                INSERT INTO mcp_servers (name, command, args, env, description, enabled, is_builtin, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)
                """,
                (server["name"], server["command"], json.dumps(server["args"]), "{}", server["description"], now, now),
            )

    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name)")
    _mark_migration_applied(cursor, "v143_mcp_hygiene", now)


def _migrate_mcp_hygiene(cursor):
    """DEPRECATED: old MCP hygiene migration preserved for external callers.

    init_db now uses _migrate_mcp_hygiene_v143 which never overwrites user edits.
    """
    import json

    now = now_cn("%Y-%m-%d %H:%M")

    # Names to purge.  "Test Server" covers frontend-created test duplicates;
    # "time-server" has zero discoverable tools and should not be demoed.
    purge_names = {"Test Server", "time-server", "calculator-server", "db-query-server"}

    # Collect ids of servers to delete so we can clean dependent tables.
    cursor.execute(
        "SELECT id, name FROM mcp_servers WHERE name IN (" + ",".join("?" * len(purge_names)) + ")",
        tuple(purge_names),
    )
    purge_rows = cursor.fetchall()
    purge_ids = {row[0] for row in purge_rows}
    purge_found_names = {row[1] for row in purge_rows}

    # Also remove any server whose name starts with "Test" or contains "test"
    # but was not matched exactly (defensive).
    cursor.execute(
        "SELECT id, name FROM mcp_servers WHERE LOWER(name) LIKE '%test%' OR LOWER(name) LIKE 'test %'"
    )
    for row in cursor.fetchall():
        purge_ids.add(row[0])
        purge_found_names.add(row[1])

    if purge_ids:
        # Cascade delete tool definitions and agent bindings for purged servers.
        cursor.execute(
            "DELETE FROM mcp_tools WHERE server_id IN (" + ",".join("?" * len(purge_ids)) + ")",
            tuple(purge_ids),
        )
        cursor.execute(
            "DELETE FROM agent_tools WHERE tool_name IN (" + ",".join("?" * len(purge_found_names)) + ")",
            tuple(purge_found_names),
        )
        cursor.execute(
            "DELETE FROM mcp_servers WHERE id IN (" + ",".join("?" * len(purge_ids)) + ")",
            tuple(purge_ids),
        )

    # Deduplicate canonical server names: keep the lowest id for each name and
    # remove duplicates that may have been created during earlier migrations/tests.
    canonical_names = ("weather-server", "workorder-server", "calendar-server")
    for name in canonical_names:
        cursor.execute(
            "SELECT id FROM mcp_servers WHERE name = ? ORDER BY id ASC",
            (name,),
        )
        rows = [r[0] for r in cursor.fetchall()]
        if len(rows) > 1:
            keep_id = rows[0]
            duplicate_ids = rows[1:]
            cursor.execute(
                "DELETE FROM mcp_tools WHERE server_id IN (" + ",".join("?" * len(duplicate_ids)) + ")",
                tuple(duplicate_ids),
            )
            cursor.execute(
                "DELETE FROM mcp_servers WHERE id IN (" + ",".join("?" * len(duplicate_ids)) + ")",
                tuple(duplicate_ids),
            )

    # Enforce unique server names to prevent duplicate canonical servers on
    # repeated seed/migration runs.
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name)"
    )

    # Canonical demo servers.  Use INSERT OR IGNORE for names, then UPDATE
    # command/args/description so re-running the migration stays idempotent.
    canonical_servers = [
        {
            "name": "weather-server",
            "command": "python",
            "args": ["/app/tools/weather_mcp_server.py"],
            "description": "天气查询 MCP Server，提供实时天气与天气建议工具。",
        },
        {
            "name": "workorder-server",
            "command": "python",
            "args": ["/app/tools/db_query_mcp_server.py"],
            "description": "工单查询 MCP Server，提供维修工单数量与待办列表只读查询。",
        },
        {
            "name": "calendar-server",
            "command": "python",
            "args": ["/app/tools/calendar_mcp_server.py"],
            "description": "日历 MCP Server，提供当前日期与预约时间相关工具。",
        },
    ]

    for server in canonical_servers:
        cursor.execute(
            """
            INSERT OR IGNORE INTO mcp_servers (name, command, args, env, description, enabled, is_builtin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)
            """,
            (server["name"], server["command"], json.dumps(server["args"]), "{}", server["description"], now, now),
        )
        cursor.execute(
            """
            UPDATE mcp_servers
            SET command = ?, args = ?, description = ?, enabled = 1, is_builtin = 1, updated_at = ?
            WHERE name = ?
            """,
            (server["command"], json.dumps(server["args"]), server["description"], now, server["name"]),
        )

    # Ensure formal demo bindings:
    # - maintenance: weather-server + workorder-server
    # - customer_service: calendar-server
    cursor.execute("SELECT agent_id FROM agents WHERE agent_id = ?", ("maintenance",))
    if cursor.fetchone():
        for tool_name in ("weather-server", "workorder-server"):
            cursor.execute(
                "INSERT OR IGNORE INTO agent_tools (agent_id, tool_name, config) VALUES (?, ?, ?)",
                ("maintenance", tool_name, "{}"),
            )
    cursor.execute("SELECT agent_id FROM agents WHERE agent_id = ?", ("customer_service",))
    if cursor.fetchone():
        cursor.execute(
            "INSERT OR IGNORE INTO agent_tools (agent_id, tool_name, config) VALUES (?, ?, ?)",
            ("customer_service", "calendar-server", "{}"),
        )


def _migrate_v1_3_observability(cursor):
    """Non-destructive migrations for V1.3 observability & cost governance."""
    now = now_cn("%Y-%m-%d %H:%M")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_traces (
            trace_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_message TEXT,
            intent TEXT,
            agent_name TEXT,
            agent_id TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    # V1.4.3: add agent_id column to existing chat_traces.
    try:
        cursor.execute("ALTER TABLE chat_traces ADD COLUMN agent_id TEXT")
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            stage TEXT,
            model_id TEXT,
            status TEXT,
            latency_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            reasoning_tokens INTEGER,
            cached_tokens INTEGER,
            total_tokens INTEGER,
            usage_source TEXT,
            model_selection_reason TEXT,
            error_summary TEXT,
            price_snapshot TEXT,
            estimated_cost_cny REAL,
            context_breakdown TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_calls_trace_id ON model_calls(trace_id)"
    )
    # V1.4.2: add context_breakdown column to existing databases.
    try:
        cursor.execute("ALTER TABLE model_calls ADD COLUMN context_breakdown TEXT")
    except sqlite3.OperationalError:
        pass

    # V1.4.3: add usage_normalized column to store uncached/cached/output split.
    try:
        cursor.execute("ALTER TABLE model_calls ADD COLUMN usage_normalized TEXT")
    except sqlite3.OperationalError:
        pass

    # V1.4.3: add invocation_mode to mcp_call_audits for policy_preinvoke traceability.
    try:
        cursor.execute("ALTER TABLE mcp_call_audits ADD COLUMN invocation_mode TEXT")
    except sqlite3.OperationalError:
        pass

    # V1.4.3: add session_id to work_orders to link confirmed orders to their chat session.
    try:
        cursor.execute("ALTER TABLE work_orders ADD COLUMN session_id TEXT")
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_call_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            server_name TEXT,
            tool_name TEXT,
            arguments TEXT,
            status TEXT,
            result_summary TEXT,
            error_summary TEXT,
            latency_ms INTEGER,
            invocation_mode TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_call_audits_trace_id ON mcp_call_audits(trace_id)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            currency TEXT DEFAULT 'CNY',
            effective_date TEXT,
            input_price_per_1m REAL,
            cached_input_price_per_1m REAL,
            output_price_per_1m REAL,
            reasoning_price_per_1m REAL,
            source_note TEXT,
            enabled INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_model_prices_model_effective ON model_prices(model_id, effective_date)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS budget_thresholds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            per_call_threshold_cny REAL,
            daily_threshold_cny REAL,
            monthly_threshold_cny REAL,
            updated_at TEXT
        )
        """
    )
    # Ensure the monthly column exists on existing databases before inserting.
    try:
        cursor.execute("ALTER TABLE budget_thresholds ADD COLUMN monthly_threshold_cny REAL")
    except sqlite3.OperationalError:
        pass
    cursor.execute(
        "INSERT OR IGNORE INTO budget_thresholds (id, per_call_threshold_cny, daily_threshold_cny, monthly_threshold_cny, updated_at) VALUES (1, NULL, NULL, NULL, ?)",
        (now,),
    )


def _migrate_v1_3_3_badcase_closure(cursor):
    """Non-destructive migrations for V1.3.3 badcase operational closure."""
    # Extend badcases with operational fields.
    for col, dtype in [
        ("source", "TEXT DEFAULT 'auto'"),
        ("original_query", "TEXT"),
        ("ai_response", "TEXT"),
        ("feedback_reason", "TEXT"),
        ("context_json", "TEXT"),
        ("trace_id", "TEXT"),
        ("priority", "TEXT DEFAULT 'medium'"),
        ("message_id", "INTEGER"),
        ("retest_response", "TEXT"),
        ("retest_context_json", "TEXT"),
        ("retest_trace_id", "TEXT"),
        ("darwin_analysis", "TEXT"),
        ("darwin_trace_id", "TEXT"),
        ("last_applied_at", "TEXT"),
        ("last_retest_at", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE badcases ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # Knowledge drafts: add knowledge_doc_id reference.
    try:
        cursor.execute("ALTER TABLE knowledge_drafts ADD COLUMN knowledge_doc_id INTEGER")
    except sqlite3.OperationalError:
        pass

    # Skill / Prompt drafts.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_prompt_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badcase_id INTEGER NOT NULL,
            skill_id INTEGER,
            skill_name TEXT,
            title TEXT NOT NULL,
            prompt_content TEXT,
            trigger_keywords TEXT,
            status TEXT DEFAULT 'draft',
            created_at TEXT,
            updated_at TEXT,
            published_at TEXT,
            published_by TEXT,
            FOREIGN KEY (badcase_id) REFERENCES badcases(id)
        )
        """
    )

    # Capability gap drafts.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS capability_gap_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badcase_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            gap_type TEXT,
            suggested_action TEXT,
            status TEXT DEFAULT 'draft',
            created_at TEXT,
            updated_at TEXT,
            accepted_at TEXT,
            accepted_by TEXT,
            FOREIGN KEY (badcase_id) REFERENCES badcases(id)
        )
        """
    )


def _migrate_v1_6_quality_governance(cursor):
    """Non-destructive schema for Badcase → Evaluation → Trace governance.

    The demo deliberately stores only operator-defined test cases and compact
    runtime evidence.  It does not pretend to provide production monitoring or
    retain a full copy of every upstream payload.
    """
    # Badcase is the operational object.  Keep symptom, expected/actual result
    # and root-cause domain separate so a label such as "answer wrong" is never
    # mistaken for a diagnosis.
    for col, dtype in [
        ("symptom", "TEXT"),
        ("expected_behavior", "TEXT"),
        ("actual_behavior", "TEXT"),
        ("root_cause_domain", "TEXT"),
        ("secondary_root_cause_domains", "TEXT"),
        ("impact_scope", "TEXT"),
        ("owner", "TEXT"),
        ("release_version", "TEXT"),
        ("release_note", "TEXT"),
        ("released_at", "TEXT"),
        ("observed_at", "TEXT"),
        ("linked_evaluation_case_id", "INTEGER"),
        ("linked_evaluation_run_id", "INTEGER"),
        ("duplicate_of_id", "INTEGER"),
        ("accepted_limitation_reason", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE badcases ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    # A Golden Set case defines business rules rather than a single frozen
    # natural-language answer.  JSON fields are kept as explicit contracts for
    # deterministic checks; qualitative judgement stays operator-owned.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_key TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            description TEXT,
            scenario TEXT,
            user_message TEXT NOT NULL,
            session_context_json TEXT,
            risk_level TEXT DEFAULT 'L2',
            expected_agent_id TEXT,
            expected_skills_json TEXT,
            expected_tools_json TEXT,
            expected_citation_docs_json TEXT,
            required_terms_json TEXT,
            forbidden_terms_json TEXT,
            expected_handoff INTEGER,
            rubric_json TEXT,
            source TEXT DEFAULT 'expert',
            source_badcase_id INTEGER,
            status TEXT DEFAULT 'draft',
            version_label TEXT,
            owner TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (source_badcase_id) REFERENCES badcases(id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_cases_status ON evaluation_cases(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_cases_badcase ON evaluation_cases(source_badcase_id)")

    # A run is immutable evidence from one explicit operator-triggered runtime
    # execution.  It links to the actual chat Trace instead of fabricating a
    # separate test result.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_case_id INTEGER NOT NULL,
            trace_id TEXT,
            session_id TEXT,
            status TEXT NOT NULL,
            answer TEXT,
            evidence_json TEXT,
            rule_results_json TEXT,
            operator_judgement TEXT,
            operator_note TEXT,
            badcase_id INTEGER,
            total_tokens INTEGER,
            estimated_cost_cny REAL,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (evaluation_case_id) REFERENCES evaluation_cases(id),
            FOREIGN KEY (badcase_id) REFERENCES badcases(id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_case ON evaluation_runs(evaluation_case_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_trace ON evaluation_runs(trace_id)")

    # Trace events are compact spans for non-model steps such as RAG and
    # handoff.  Tool calls remain in the existing detailed MCP audit table.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            span_name TEXT NOT NULL,
            status TEXT,
            latency_ms INTEGER,
            input_summary TEXT,
            output_summary TEXT,
            metadata_json TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_events_trace ON trace_events(trace_id)")

    # Associate normal chat traces with an optional evaluation run without
    # changing how ordinary owner conversations are stored.
    for col, dtype in [
        ("run_type", "TEXT DEFAULT 'chat'"),
        ("evaluation_case_id", "INTEGER"),
        ("evaluation_run_id", "INTEGER"),
        ("risk_level", "TEXT"),
        ("version_snapshot", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE chat_traces ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass


def _migrate_v1_8_runtime_convergence(cursor):
    """Add the V1.8 control-plane and evidence-plane schema.

    Platform CRUD tables remain the editable source configuration.  A runtime
    can only execute a validated, published release compiled from those
    tables, and each conversation pins one immutable release snapshot.
    """
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_releases (
            release_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL UNIQUE,
            status TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            config_json TEXT NOT NULL,
            validation_json TEXT,
            parent_release_id TEXT,
            created_by TEXT,
            created_at TEXT NOT NULL,
            published_at TEXT,
            superseded_at TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_runtime_releases_status ON runtime_releases(status, version)"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_release_pointer (
            pointer_key TEXT PRIMARY KEY,
            release_id TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (release_id) REFERENCES runtime_releases(release_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_config_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL UNIQUE,
            release_id TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (release_id) REFERENCES runtime_releases(release_id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_run_snapshots_release ON run_config_snapshots(release_id)"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            release_id TEXT NOT NULL,
            server_id INTEGER,
            server_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            effect TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            allowed_paths_json TEXT NOT NULL,
            requires_confirmation INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            policy_reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(release_id, server_name, tool_name),
            FOREIGN KEY (release_id) REFERENCES runtime_releases(release_id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_policies_release ON tool_policies(release_id)"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_evidence_ledgers (
            trace_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            release_id TEXT,
            config_hash TEXT,
            runtime_path TEXT,
            status TEXT,
            ledger_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_evidence_ledgers_session ON run_evidence_ledgers(session_id, created_at)"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS action_proposals (
            proposal_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            trace_id TEXT,
            release_id TEXT,
            action_type TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_action_proposals_session ON action_proposals(session_id, created_at)"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS action_approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposal_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            actor TEXT NOT NULL,
            comment TEXT,
            decided_at TEXT NOT NULL,
            FOREIGN KEY (proposal_id) REFERENCES action_proposals(proposal_id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_action_approvals_proposal ON action_approvals(proposal_id, id)"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS action_receipts (
            receipt_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE,
            idempotency_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            resource_type TEXT,
            resource_id TEXT,
            result_json TEXT,
            committed_at TEXT,
            error_summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (proposal_id) REFERENCES action_proposals(proposal_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_knowledge_bindings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            knowledge_doc_id INTEGER NOT NULL,
            enabled INTEGER DEFAULT 1,
            UNIQUE(agent_id, knowledge_doc_id),
            FOREIGN KEY (knowledge_doc_id) REFERENCES knowledge_docs(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_knowledge_scopes (
            agent_id TEXT PRIMARY KEY,
            binding_mode TEXT NOT NULL DEFAULT 'explicit',
            updated_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_acceptance_runs (
            acceptance_run_id TEXT PRIMARY KEY,
            case_key TEXT NOT NULL,
            release_id TEXT,
            status TEXT NOT NULL,
            evidence_json TEXT,
            cleanup_json TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )

def create_work_order(
    work_order_id: str,
    room_id: str,
    issue_type: str,
    issue_desc: str,
    urgency: str,
    contact_name: str,
    contact_phone: str,
    appointment_time: str,
    status: str = "待派单",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO work_orders
        (id, session_id, room_id, contact_name, contact_phone, issue_type, issue_desc, urgency, status,
         appointment_time, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (work_order_id, session_id, room_id, contact_name, contact_phone, issue_type, issue_desc, urgency, status, appointment_time, now, now),
    )
    conn.commit()
    conn.close()
    return get_work_order(work_order_id)


# ---------------------------------------------------------------------------
# Work order drafts (V1.4.3): pending explicit user confirmation per session.
# ---------------------------------------------------------------------------
def save_work_order_draft(
    session_id: str,
    room_id: str,
    issue_type: str,
    issue_desc: str,
    urgency: str,
    contact_name: str,
    contact_phone: str,
    appointment_time: str,
) -> Dict[str, Any]:
    """Upsert a work-order draft for the session. Does not create a real work order."""
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO work_order_drafts
        (session_id, room_id, issue_type, issue_desc, urgency, contact_name, contact_phone, appointment_time, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            room_id = excluded.room_id,
            issue_type = excluded.issue_type,
            issue_desc = excluded.issue_desc,
            urgency = excluded.urgency,
            contact_name = excluded.contact_name,
            contact_phone = excluded.contact_phone,
            appointment_time = excluded.appointment_time,
            updated_at = excluded.updated_at
        """,
        (session_id, room_id, issue_type, issue_desc, urgency, contact_name, contact_phone, appointment_time, now, now),
    )
    conn.commit()
    conn.close()
    return get_work_order_draft(session_id)


def get_work_order_draft(session_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM work_order_drafts WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_work_order_draft(session_id: str) -> None:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM work_order_drafts WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def get_work_order(work_order_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM work_orders WHERE id = ?", (work_order_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_work_orders(
    status: Optional[str] = None,
    room_id: Optional[str] = None,
    date_prefix: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM work_orders WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if room_id:
        query += " AND room_id = ?"
        params.append(room_id)
    if date_prefix:
        query += " AND id LIKE ?"
        params.append(f"WO-{date_prefix}-%")
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_work_order_status(
    work_order_id: str,
    status: str,
    assigned_to: Optional[str] = None,
    completion_note: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()

    fields = ["status = ?", "updated_at = ?"]
    params = [status, now]
    if assigned_to is not None:
        fields.append("assigned_to = ?")
        params.append(assigned_to)
    if completion_note is not None:
        fields.append("completion_note = ?")
        params.append(completion_note)

    params.append(work_order_id)
    cursor.execute(
        f"UPDATE work_orders SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_work_order(work_order_id)


# -----------------------------------------------------------------------------
# Knowledge docs CRUD
# -----------------------------------------------------------------------------


def create_knowledge_doc(
    title: str,
    content: str,
    category: str,
    source_type: str = "business",
    index_status: str = "pending",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    split_strategy: str = "auto",
) -> Dict[str, Any]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO knowledge_docs
        (title, content, category, source_type, is_indexed, index_status, chunk_size, chunk_overlap, split_strategy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (title, content, category, source_type, 1 if source_type == "business" else 0, index_status, chunk_size, chunk_overlap, split_strategy),
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_knowledge_doc(doc_id)


def get_knowledge_doc(doc_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_docs WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_knowledge_docs() -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_docs ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_knowledge_doc(
    doc_id: int,
    title: str,
    content: str,
    category: str,
    source_type: Optional[str] = None,
    index_status: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    split_strategy: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()

    fields = ["title = ?", "content = ?", "category = ?"]
    params = [title, content, category]
    if source_type is not None:
        fields.append("source_type = ?")
        params.append(source_type)
        fields.append("is_indexed = ?")
        params.append(1 if source_type == "business" else 0)
    if index_status is not None:
        fields.append("index_status = ?")
        params.append(index_status)
    if chunk_size is not None:
        fields.append("chunk_size = ?")
        params.append(chunk_size)
    if chunk_overlap is not None:
        fields.append("chunk_overlap = ?")
        params.append(chunk_overlap)
    if split_strategy is not None:
        fields.append("split_strategy = ?")
        params.append(split_strategy)

    params.append(doc_id)
    cursor.execute(
        f"UPDATE knowledge_docs SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_knowledge_doc(doc_id)


def set_knowledge_doc_indexed(doc_id: int, index_status: str, chunk_count: int):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE knowledge_docs SET index_status = ?, chunk_count = ? WHERE id = ?",
        (index_status, chunk_count, doc_id),
    )
    conn.commit()
    conn.close()


def set_knowledge_doc_indexed_flag(doc_id: int, is_indexed: bool):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE knowledge_docs SET is_indexed = ? WHERE id = ?",
        (1 if is_indexed else 0, doc_id),
    )
    conn.commit()
    conn.close()


def delete_knowledge_doc(doc_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge_docs WHERE id = ?", (doc_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def search_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Simple keyword search over knowledge docs."""
    rows = list_knowledge_docs()
    query_terms = [q for q in query.split() if q]
    scored = []
    for row in rows:
        if not row.get("is_indexed"):
            continue
        text = f"{row['title']} {row['content']}".lower()
        score = sum(1 for q in query_terms if q.lower() in text)
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


# -----------------------------------------------------------------------------
# Badcases CRUD
# -----------------------------------------------------------------------------


def create_badcase(
    title: str,
    description: str,
    category: str = "other",
    status: str = "pending",
    created_at: Optional[str] = None,
    evidence: Optional[str] = None,
    source_message_id: Optional[int] = None,
    session_id: Optional[str] = None,
    root_cause: Optional[str] = None,
    fix_plan: Optional[str] = None,
    source: str = "auto",
    original_query: Optional[str] = None,
    ai_response: Optional[str] = None,
    feedback_reason: Optional[str] = None,
    context_json: Optional[str] = None,
    trace_id: Optional[str] = None,
    priority: str = "medium",
    message_id: Optional[int] = None,
    symptom: Optional[str] = None,
    expected_behavior: Optional[str] = None,
    actual_behavior: Optional[str] = None,
    root_cause_domain: Optional[str] = None,
    secondary_root_cause_domains: Optional[str] = None,
    impact_scope: Optional[str] = None,
    owner: Optional[str] = None,
    linked_evaluation_case_id: Optional[int] = None,
    linked_evaluation_run_id: Optional[int] = None,
) -> Dict[str, Any]:
    now = created_at or now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO badcases
        (title, description, category, status, created_at, evidence, source_message_id, session_id,
         root_cause, fix_plan, source, original_query, ai_response, feedback_reason, context_json,
         trace_id, priority, message_id, symptom, expected_behavior, actual_behavior,
         root_cause_domain, secondary_root_cause_domains, impact_scope, owner,
         linked_evaluation_case_id, linked_evaluation_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (title, description, category, status, now, evidence, source_message_id, session_id,
         root_cause, fix_plan, source, original_query, ai_response, feedback_reason, context_json,
         trace_id, priority, message_id, symptom, expected_behavior, actual_behavior,
         root_cause_domain, secondary_root_cause_domains, impact_scope, owner,
         linked_evaluation_case_id, linked_evaluation_run_id),
    )
    case_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_badcase(case_id)


def get_badcase(case_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM badcases WHERE id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_badcases(
    status: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    has_trace: Optional[bool] = None,
    has_retest: Optional[bool] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM badcases WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if category:
        query += " AND category = ?"
        params.append(category)
    if source:
        query += " AND source = ?"
        params.append(source)
    if has_trace is True:
        query += " AND darwin_trace_id IS NOT NULL AND darwin_trace_id != ''"
    elif has_trace is False:
        query += " AND (darwin_trace_id IS NULL OR darwin_trace_id = '')"
    if has_retest is True:
        query += " AND retest_response IS NOT NULL AND retest_response != ''"
    elif has_retest is False:
        query += " AND (retest_response IS NULL OR retest_response = '')"
    if created_after:
        query += " AND created_at >= ?"
        params.append(created_after)
    if created_before:
        query += " AND created_at <= ?"
        params.append(created_before)
    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_badcase(
    case_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    evidence: Optional[str] = None,
    root_cause: Optional[str] = None,
    fix_plan: Optional[str] = None,
    verified_by: Optional[str] = None,
    rejected_reason: Optional[str] = None,
    source: Optional[str] = None,
    original_query: Optional[str] = None,
    ai_response: Optional[str] = None,
    feedback_reason: Optional[str] = None,
    context_json: Optional[str] = None,
    trace_id: Optional[str] = None,
    priority: Optional[str] = None,
    message_id: Optional[int] = None,
    retest_response: Optional[str] = None,
    retest_context_json: Optional[str] = None,
    retest_trace_id: Optional[str] = None,
    darwin_analysis: Optional[str] = None,
    darwin_trace_id: Optional[str] = None,
    last_applied_at: Optional[str] = None,
    last_retest_at: Optional[str] = None,
    symptom: Optional[str] = None,
    expected_behavior: Optional[str] = None,
    actual_behavior: Optional[str] = None,
    root_cause_domain: Optional[str] = None,
    secondary_root_cause_domains: Optional[str] = None,
    impact_scope: Optional[str] = None,
    owner: Optional[str] = None,
    release_version: Optional[str] = None,
    release_note: Optional[str] = None,
    released_at: Optional[str] = None,
    observed_at: Optional[str] = None,
    linked_evaluation_case_id: Optional[int] = None,
    linked_evaluation_run_id: Optional[int] = None,
    duplicate_of_id: Optional[int] = None,
    accepted_limitation_reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()

    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [
        ("title", title),
        ("description", description),
        ("category", category),
        ("status", status),
        ("evidence", evidence),
        ("root_cause", root_cause),
        ("fix_plan", fix_plan),
        ("verified_by", verified_by),
        ("rejected_reason", rejected_reason),
        ("source", source),
        ("original_query", original_query),
        ("ai_response", ai_response),
        ("feedback_reason", feedback_reason),
        ("context_json", context_json),
        ("trace_id", trace_id),
        ("priority", priority),
        ("message_id", message_id),
        ("retest_response", retest_response),
        ("retest_context_json", retest_context_json),
        ("retest_trace_id", retest_trace_id),
        ("darwin_analysis", darwin_analysis),
        ("darwin_trace_id", darwin_trace_id),
        ("last_applied_at", last_applied_at),
        ("last_retest_at", last_retest_at),
        ("symptom", symptom),
        ("expected_behavior", expected_behavior),
        ("actual_behavior", actual_behavior),
        ("root_cause_domain", root_cause_domain),
        ("secondary_root_cause_domains", secondary_root_cause_domains),
        ("impact_scope", impact_scope),
        ("owner", owner),
        ("release_version", release_version),
        ("release_note", release_note),
        ("released_at", released_at),
        ("observed_at", observed_at),
        ("linked_evaluation_case_id", linked_evaluation_case_id),
        ("linked_evaluation_run_id", linked_evaluation_run_id),
        ("duplicate_of_id", duplicate_of_id),
        ("accepted_limitation_reason", accepted_limitation_reason),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)

    if status == "closed":
        fields.append("closed_at = ?")
        params.append(now)
    if status == "verifying" and verified_by:
        fields.append("verified_at = ?")
        params.append(now)

    params.append(case_id)
    cursor.execute(
        f"UPDATE badcases SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_badcase(case_id)


def delete_badcase(case_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM badcases WHERE id = ?", (case_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ---------------------------------------------------------------------------
# Badcase Actions
# ---------------------------------------------------------------------------


def add_badcase_action(
    badcase_id: int,
    action_type: str,
    action_detail: Optional[str] = None,
    status_before: Optional[str] = None,
    status_after: Optional[str] = None,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO badcase_actions
        (badcase_id, action_type, action_detail, status_before, status_after, created_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (badcase_id, action_type, action_detail, status_before, status_after, created_by, now),
    )
    action_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_badcase_action(action_id)


def get_badcase_action(action_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM badcase_actions WHERE id = ?", (action_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_badcase_actions(badcase_id: int) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM badcase_actions WHERE badcase_id = ? ORDER BY id ASC",
        (badcase_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_badcase_id_by_trace_id(trace_id: str) -> Optional[int]:
    """Return the badcase id linked to a trace/darwin/retest trace id, if any."""
    if not trace_id:
        return None
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id FROM badcases
        WHERE trace_id = ? OR darwin_trace_id = ? OR retest_trace_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (trace_id, trace_id, trace_id),
    )
    row = cursor.fetchone()
    conn.close()
    return row["id"] if row else None


# -----------------------------------------------------------------------------
# Evaluation / Golden Set CRUD (V1.6)
# -----------------------------------------------------------------------------


def _json_text(value: Any) -> str:
    """Serialize optional structured fields consistently for SQLite."""
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else [], ensure_ascii=False, default=str)


def _parse_json_text(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _enrich_evaluation_case(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    for field, default in [
        ("session_context_json", {}),
        ("expected_skills_json", []),
        ("expected_tools_json", []),
        ("expected_citation_docs_json", []),
        ("required_terms_json", []),
        ("forbidden_terms_json", []),
        ("rubric_json", {}),
    ]:
        item[field[:-5]] = _parse_json_text(item.get(field), default)
    if item.get("expected_handoff") is not None:
        item["expected_handoff"] = bool(item.get("expected_handoff"))
    return item


def create_evaluation_case(
    case_key: str,
    title: str,
    user_message: str,
    description: str = "",
    scenario: str = "",
    session_context: Optional[Dict[str, Any]] = None,
    risk_level: str = "L2",
    expected_agent_id: Optional[str] = None,
    expected_skills: Optional[List[str]] = None,
    expected_tools: Optional[List[str]] = None,
    expected_citation_docs: Optional[List[str]] = None,
    required_terms: Optional[List[str]] = None,
    forbidden_terms: Optional[List[str]] = None,
    expected_handoff: Optional[bool] = None,
    rubric: Optional[Dict[str, Any]] = None,
    source: str = "expert",
    source_badcase_id: Optional[int] = None,
    status: str = "draft",
    version_label: Optional[str] = None,
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO evaluation_cases
        (case_key, title, description, scenario, user_message, session_context_json, risk_level,
         expected_agent_id, expected_skills_json, expected_tools_json, expected_citation_docs_json,
         required_terms_json, forbidden_terms_json, expected_handoff, rubric_json, source,
         source_badcase_id, status, version_label, owner, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_key, title, description, scenario, user_message, _json_text(session_context or {}), risk_level,
            expected_agent_id, _json_text(expected_skills or []), _json_text(expected_tools or []),
            _json_text(expected_citation_docs or []), _json_text(required_terms or []),
            _json_text(forbidden_terms or []), None if expected_handoff is None else int(expected_handoff),
            _json_text(rubric or {}), source, source_badcase_id, status, version_label, owner, now, now,
        ),
    )
    case_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_evaluation_case(case_id) or {}


def get_evaluation_case(case_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluation_cases WHERE id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    return _enrich_evaluation_case(dict(row) if row else None)


def list_evaluation_cases(status: Optional[str] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    sql = "SELECT * FROM evaluation_cases WHERE 1=1"
    params: List[Any] = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if source:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY updated_at DESC, id DESC"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return [_enrich_evaluation_case(dict(row)) or {} for row in rows]


def update_evaluation_case(case_id: int, **updates: Any) -> Optional[Dict[str, Any]]:
    allowed = {
        "case_key", "title", "description", "scenario", "user_message", "risk_level",
        "expected_agent_id", "expected_handoff", "source", "source_badcase_id", "status",
        "version_label", "owner",
    }
    json_fields = {
        "session_context": "session_context_json",
        "expected_skills": "expected_skills_json",
        "expected_tools": "expected_tools_json",
        "expected_citation_docs": "expected_citation_docs_json",
        "required_terms": "required_terms_json",
        "forbidden_terms": "forbidden_terms_json",
        "rubric": "rubric_json",
    }
    fields = ["updated_at = ?"]
    params: List[Any] = [now_cn()]
    for key, value in updates.items():
        if value is None:
            continue
        if key in json_fields:
            fields.append(f"{json_fields[key]} = ?")
            params.append(_json_text(value))
        elif key in allowed:
            fields.append(f"{key} = ?")
            params.append(int(value) if key == "expected_handoff" and value is not None else value)
    if len(fields) == 1:
        return get_evaluation_case(case_id)
    params.append(case_id)
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE evaluation_cases SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    return get_evaluation_case(case_id)


def delete_evaluation_case(case_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM evaluation_cases WHERE id = ?", (case_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def _enrich_evaluation_run(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["evidence"] = _parse_json_text(item.get("evidence_json"), {})
    item["rule_results"] = _parse_json_text(item.get("rule_results_json"), [])
    return item


def create_evaluation_run(
    evaluation_case_id: int,
    status: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    answer: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    rule_results: Optional[List[Dict[str, Any]]] = None,
    total_tokens: Optional[int] = None,
    estimated_cost_cny: Optional[float] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO evaluation_runs
        (evaluation_case_id, trace_id, session_id, status, answer, evidence_json, rule_results_json,
         total_tokens, estimated_cost_cny, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (evaluation_case_id, trace_id, session_id, status, answer, _json_text(evidence or {}),
         _json_text(rule_results or []), total_tokens, estimated_cost_cny, now, now),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_evaluation_run(run_id) or {}


def get_evaluation_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    return _enrich_evaluation_run(dict(row) if row else None)


def get_evaluation_run_by_trace_id(trace_id: str) -> Optional[Dict[str, Any]]:
    if not trace_id:
        return None
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluation_runs WHERE trace_id = ? ORDER BY id DESC LIMIT 1", (trace_id,))
    row = cursor.fetchone()
    conn.close()
    return _enrich_evaluation_run(dict(row) if row else None)


def list_evaluation_runs(evaluation_case_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    sql = "SELECT * FROM evaluation_runs"
    params: List[Any] = []
    if evaluation_case_id is not None:
        sql += " WHERE evaluation_case_id = ?"
        params.append(evaluation_case_id)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return [_enrich_evaluation_run(dict(row)) or {} for row in rows]


def update_evaluation_run(
    run_id: int,
    status: Optional[str] = None,
    operator_judgement: Optional[str] = None,
    operator_note: Optional[str] = None,
    badcase_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    fields = ["updated_at = ?"]
    params: List[Any] = [now_cn()]
    for col, value in [
        ("status", status),
        ("operator_judgement", operator_judgement),
        ("operator_note", operator_note),
        ("badcase_id", badcase_id),
    ]:
        if value is not None:
            fields.append(f"{col} = ?")
            params.append(value)
    params.append(run_id)
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE evaluation_runs SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    return get_evaluation_run(run_id)


def evaluation_summary() -> Dict[str, Any]:
    """Return a compact quality/cost summary without inventing production KPIs."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) AS passed,
               SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN status = 'needs_manual_review' THEN 1 ELSE 0 END) AS manual_review,
               COALESCE(SUM(estimated_cost_cny), 0) AS total_cost,
               COALESCE(SUM(CASE WHEN status = 'passed' THEN estimated_cost_cny ELSE 0 END), 0) AS passed_cost
        FROM evaluation_runs
        """
    )
    row = dict(cursor.fetchone() or {})
    cursor.execute(
        "SELECT risk_level, COUNT(*) AS total FROM evaluation_cases WHERE status = 'active' GROUP BY risk_level"
    )
    by_risk = {r["risk_level"]: r["total"] for r in cursor.fetchall()}
    conn.close()
    passed = int(row.get("passed") or 0)
    total = int(row.get("total") or 0)
    total_cost = float(row.get("total_cost") or 0.0)
    return {
        "cases_active_by_risk": by_risk,
        "runs_total": total,
        "runs_passed": passed,
        "runs_failed": int(row.get("failed") or 0),
        "runs_needing_manual_review": int(row.get("manual_review") or 0),
        "deterministic_pass_rate": round((passed / total) * 100, 2) if total else None,
        "model_direct_cost_cny": round(total_cost, 8),
        "cost_per_passed_run_cny": round(total_cost / passed, 8) if passed else None,
        "note": "仅统计已显式运行的演示评估与模型直接 Token 估算；不代表生产 SLA 或全量业务成本。",
    }


# -----------------------------------------------------------------------------
# Compact trace event spans (V1.6)
# -----------------------------------------------------------------------------


def record_trace_event(
    trace_id: str,
    span_name: str,
    status: str = "success",
    latency_ms: Optional[int] = None,
    input_summary: Optional[str] = None,
    output_summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO trace_events
        (trace_id, span_name, status, latency_ms, input_summary, output_summary, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trace_id, span_name, status, latency_ms, input_summary, output_summary,
         _json_text(metadata or {}), now_cn()),
    )
    event_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM trace_events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    conn.close()
    event = dict(row) if row else {}
    event["metadata"] = _parse_json_text(event.get("metadata_json"), {})
    return event


def list_trace_events(trace_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trace_events WHERE trace_id = ? ORDER BY id ASC", (trace_id,))
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _parse_json_text(item.get("metadata_json"), {})
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Knowledge Drafts
# ---------------------------------------------------------------------------


def create_knowledge_draft(
    badcase_id: Optional[int],
    title: str,
    content: str,
    category: str = "未分类",
    status: str = "draft",
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO knowledge_drafts
        (badcase_id, title, content, category, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (badcase_id, title, content, category, status, now, now),
    )
    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_knowledge_draft(draft_id)


def get_knowledge_draft(draft_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_drafts WHERE id = ?", (draft_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_knowledge_drafts(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM knowledge_drafts WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_knowledge_draft(
    draft_id: int,
    title: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    knowledge_doc_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [("title", title), ("content", content), ("category", category), ("status", status), ("knowledge_doc_id", knowledge_doc_id)]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    params.append(draft_id)
    cursor.execute(
        f"UPDATE knowledge_drafts SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_knowledge_draft(draft_id)


def delete_knowledge_draft(draft_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge_drafts WHERE id = ?", (draft_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# -----------------------------------------------------------------------------
# Skill / Prompt Drafts
# -----------------------------------------------------------------------------

def create_skill_prompt_draft(
    badcase_id: int,
    title: str,
    skill_name: str,
    prompt_content: str,
    trigger_keywords: Optional[str] = None,
    skill_id: Optional[int] = None,
    status: str = "draft",
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO skill_prompt_drafts
        (badcase_id, skill_id, skill_name, title, prompt_content, trigger_keywords, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (badcase_id, skill_id, skill_name, title, prompt_content, trigger_keywords, status, now, now),
    )
    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_skill_prompt_draft(draft_id)


def get_skill_prompt_draft(draft_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skill_prompt_drafts WHERE id = ?", (draft_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_skill_prompt_drafts(
    badcase_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM skill_prompt_drafts WHERE 1=1"
    params = []
    if badcase_id is not None:
        query += " AND badcase_id = ?"
        params.append(badcase_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_skill_prompt_draft(
    draft_id: int,
    title: Optional[str] = None,
    skill_name: Optional[str] = None,
    prompt_content: Optional[str] = None,
    trigger_keywords: Optional[str] = None,
    skill_id: Optional[int] = None,
    status: Optional[str] = None,
    published_at: Optional[str] = None,
    published_by: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [
        ("title", title),
        ("skill_name", skill_name),
        ("prompt_content", prompt_content),
        ("trigger_keywords", trigger_keywords),
        ("skill_id", skill_id),
        ("status", status),
        ("published_at", published_at),
        ("published_by", published_by),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    params.append(draft_id)
    cursor.execute(
        f"UPDATE skill_prompt_drafts SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_skill_prompt_draft(draft_id)


def delete_skill_prompt_draft(draft_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM skill_prompt_drafts WHERE id = ?", (draft_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# -----------------------------------------------------------------------------
# Capability Gap Drafts
# -----------------------------------------------------------------------------

def create_capability_gap_draft(
    badcase_id: int,
    title: str,
    description: str,
    gap_type: str,
    suggested_action: Optional[str] = None,
    status: str = "draft",
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO capability_gap_drafts
        (badcase_id, title, description, gap_type, suggested_action, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (badcase_id, title, description, gap_type, suggested_action, status, now, now),
    )
    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_capability_gap_draft(draft_id)


def get_capability_gap_draft(draft_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM capability_gap_drafts WHERE id = ?", (draft_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_capability_gap_drafts(
    badcase_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM capability_gap_drafts WHERE 1=1"
    params = []
    if badcase_id is not None:
        query += " AND badcase_id = ?"
        params.append(badcase_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_capability_gap_draft(
    draft_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gap_type: Optional[str] = None,
    suggested_action: Optional[str] = None,
    status: Optional[str] = None,
    accepted_at: Optional[str] = None,
    accepted_by: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [
        ("title", title),
        ("description", description),
        ("gap_type", gap_type),
        ("suggested_action", suggested_action),
        ("status", status),
        ("accepted_at", accepted_at),
        ("accepted_by", accepted_by),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    params.append(draft_id)
    cursor.execute(
        f"UPDATE capability_gap_drafts SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_capability_gap_draft(draft_id)


def delete_capability_gap_draft(draft_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM capability_gap_drafts WHERE id = ?", (draft_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# -----------------------------------------------------------------------------
# Skills CRUD
# -----------------------------------------------------------------------------


def create_skill(
    name: str,
    description: str,
    instructions: str,
    category: str,
    enabled: bool = True,
    trigger_condition: str = "",
    skill_metadata: Optional[Dict[str, Any]] = None,
    storage_path: str = "",
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO skills
        (name, description, instructions, category, enabled, trigger_condition, skill_metadata, storage_path, model_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            description,
            instructions,
            category,
            1 if enabled else 0,
            trigger_condition,
            json.dumps(skill_metadata, ensure_ascii=False) if skill_metadata else None,
            storage_path,
            model_id,
            now,
            now,
        ),
    )
    skill_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_skill(skill_id)


def _parse_skill_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("skill_metadata")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def get_skill(skill_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skills WHERE id = ?", (skill_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    row = _row_with_bool(row, "enabled")
    row["skill_metadata"] = _parse_skill_metadata(row)
    return row


def get_skill_by_name(name: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skills WHERE name = ? LIMIT 1", (name,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    row = _row_with_bool(row, "enabled")
    row["skill_metadata"] = _parse_skill_metadata(row)
    return row


def list_skills() -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skills ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        r = _row_with_bool(r, "enabled")
        r["skill_metadata"] = _parse_skill_metadata(r)
        result.append(r)
    return result


def update_skill(
    skill_id: int,
    name: str,
    description: str,
    instructions: str,
    category: str,
    enabled: bool,
    trigger_condition: str = "",
    skill_metadata: Optional[Dict[str, Any]] = None,
    storage_path: str = "",
    model_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE skills SET
            name = ?, description = ?, instructions = ?, category = ?, enabled = ?,
            trigger_condition = ?, skill_metadata = ?, storage_path = ?, model_id = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            name,
            description,
            instructions,
            category,
            1 if enabled else 0,
            trigger_condition,
            json.dumps(skill_metadata, ensure_ascii=False) if skill_metadata else None,
            storage_path,
            model_id,
            now,
            skill_id,
        ),
    )
    conn.commit()
    conn.close()
    return get_skill(skill_id)


def delete_skill(skill_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM skill_versions WHERE skill_id = ?", (skill_id,))
    cursor.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def _skill_snapshot(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Return the persisted fields that define a Skill release."""
    return {
        "id": skill.get("id"),
        "name": skill.get("name", ""),
        "description": skill.get("description", ""),
        "instructions": skill.get("instructions", ""),
        "category": skill.get("category", ""),
        "enabled": bool(skill.get("enabled")),
        "trigger_condition": skill.get("trigger_condition", ""),
        "skill_metadata": skill.get("skill_metadata") or {},
        "storage_path": skill.get("storage_path", ""),
        "model_id": skill.get("model_id"),
    }


def create_skill_version(
    skill_id: int,
    version: str,
    change_summary: str = "",
    created_by: str = "平台管理员",
) -> Optional[Dict[str, Any]]:
    """Persist one immutable Skill release snapshot if it does not exist."""
    skill = get_skill(skill_id)
    if not skill:
        return None
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO skill_versions
        (skill_id, version, snapshot_json, change_summary, created_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            skill_id,
            version,
            json.dumps(_skill_snapshot(skill), ensure_ascii=False),
            change_summary,
            created_by,
            now_cn(),
        ),
    )
    conn.commit()
    conn.close()
    return get_skill_version(skill_id, version)


def get_skill_version(skill_id: int, version: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
        (skill_id, version),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    try:
        result["snapshot"] = json.loads(result.pop("snapshot_json") or "{}")
    except Exception:
        result["snapshot"] = {}
    return result


def list_skill_versions(skill_id: int) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM skill_versions WHERE skill_id = ? ORDER BY id DESC",
        (skill_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        item = dict(row)
        try:
            snapshot = json.loads(item.pop("snapshot_json") or "{}")
        except Exception:
            snapshot = {}
        # The list is an audit index; detailed content is available only when
        # the operator explicitly asks to restore a version.
        item["snapshot_summary"] = {
            "name": snapshot.get("name", ""),
            "trigger_condition": snapshot.get("trigger_condition", ""),
            "enabled": snapshot.get("enabled"),
        }
        result.append(item)
    return result


def _migrate_skill_governance_v152(cursor):
    """Write only baseline audit records for legacy Skills, once."""
    key = "v152_skill_version_baselines"
    cursor.execute("SELECT 1 FROM migration_meta WHERE key = ?", (key,))
    if cursor.fetchone():
        return
    cursor.execute("SELECT * FROM skills")
    for row in cursor.fetchall():
        skill = dict(row)
        raw_metadata = skill.get("skill_metadata")
        try:
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        except Exception:
            metadata = {}
        version = str(metadata.get("version") or "legacy-1.0.0")
        snapshot = {
            "id": skill.get("id"),
            "name": skill.get("name", ""),
            "description": skill.get("description", ""),
            "instructions": skill.get("instructions", ""),
            "category": skill.get("category", ""),
            "enabled": bool(skill.get("enabled")),
            "trigger_condition": skill.get("trigger_condition", ""),
            "skill_metadata": metadata,
            "storage_path": skill.get("storage_path", ""),
            "model_id": skill.get("model_id"),
        }
        cursor.execute(
            """INSERT OR IGNORE INTO skill_versions
               (skill_id, version, snapshot_json, change_summary, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (skill["id"], version, json.dumps(snapshot, ensure_ascii=False), "V1.5.2 治理基线", "系统迁移", now_cn()),
        )
    _mark_migration_applied(cursor, key, now_cn())


# -----------------------------------------------------------------------------
# MCP Servers CRUD
# -----------------------------------------------------------------------------


def create_mcp_server(
    name: str,
    command: str,
    args: Optional[List[str]],
    env: Optional[Dict[str, str]],
    description: str,
    enabled: bool = True,
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO mcp_servers (name, command, args, env, description, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            name,
            command,
            json.dumps(args) if args is not None else None,
            json.dumps(env) if env is not None else None,
            description,
            1 if enabled else 0,
            now,
            now,
        ),
    )
    server_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_mcp_server(server_id)


def get_mcp_server(server_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mcp_servers WHERE id = ?", (server_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_json_fields(row, "enabled", ["args", "env"], {"args": [], "env": {}}) if row else None


def list_mcp_servers() -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mcp_servers ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [_parse_json_fields(r, "enabled", ["args", "env"], {"args": [], "env": {}}) for r in rows]


def update_mcp_server(
    server_id: int,
    name: str,
    command: str,
    args: Optional[List[str]],
    env: Optional[Dict[str, str]],
    description: str,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE mcp_servers SET name = ?, command = ?, args = ?, env = ?, description = ?, enabled = ?, updated_at = ? WHERE id = ?",
        (
            name,
            command,
            json.dumps(args) if args is not None else None,
            json.dumps(env) if env is not None else None,
            description,
            1 if enabled else 0,
            now,
            server_id,
        ),
    )
    conn.commit()
    conn.close()
    return get_mcp_server(server_id)


def delete_mcp_server(server_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def _row_with_bool(row: sqlite3.Row, field: str) -> Dict[str, Any]:
    """Convert 0/1 integer to boolean for named field."""
    d = dict(row)
    if field in d:
        d[field] = bool(d[field])
    return d


def _parse_json_fields(row: sqlite3.Row, bool_field: str, json_fields: List[str], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Convert boolean field and parse JSON strings for given fields."""
    d = _row_with_bool(row, bool_field)
    for f in json_fields:
        if f in d:
            try:
                d[f] = json.loads(d[f]) if d[f] is not None else defaults.get(f)
            except (json.JSONDecodeError, TypeError):
                d[f] = defaults.get(f)
    return d


# Chat Messages
# -----------------------------------------------------------------------------


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    token_count: int = 0,
    round_token_count: Optional[int] = None,
    token_detail: Optional[Dict[str, Any]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    activated_skills: Optional[List[str]] = None,
    route_intent: Optional[str] = None,
    route_reason: Optional[str] = None,
    current_agent: Optional[str] = None,
    current_agent_id: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    model_id: Optional[str] = None,
    thinking_enabled: Optional[bool] = None,
    model_selection_reason: Optional[str] = None,
    trace_id: Optional[str] = None,
    status: Optional[str] = None,
    latency_ms: Optional[int] = None,
    error_summary: Optional[str] = None,
    mcp_calls: Optional[List[Dict[str, Any]]] = None,
    usage_source: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_messages (
            session_id, role, content, token_count, round_token_count, token_detail, citations, activated_skills,
            route_intent, route_reason, current_agent, current_agent_id, tool_calls, model_id, thinking_enabled,
            model_selection_reason, trace_id, status, latency_ms, error_summary, mcp_calls, usage_source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            token_count,
            round_token_count if round_token_count is not None else token_count,
            json.dumps(token_detail) if token_detail else None,
            json.dumps(citations) if citations else None,
            json.dumps(activated_skills) if activated_skills else None,
            route_intent,
            route_reason,
            current_agent,
            current_agent_id,
            json.dumps(tool_calls) if tool_calls else None,
            model_id,
            1 if thinking_enabled else 0,
            model_selection_reason,
            trace_id,
            status,
            latency_ms,
            error_summary,
            json.dumps(mcp_calls) if mcp_calls else None,
            usage_source,
            now,
        ),
    )
    message_id = cursor.lastrowid

    # Update session metadata for owner messages and assistant messages.
    if role in ("user", "owner", "assistant", "staff"):
        preview = (content or "")[:80]
        title_update = ""
        if role in ("user", "owner"):
            cursor.execute("SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,))
            existing_title = cursor.fetchone()
            if not existing_title or not existing_title[0]:
                title = (content or "").strip().replace("\n", " ")[:30]
                title_update = ", title = ?"
                title_args = (title,)
            else:
                title_args = ()
        else:
            title_args = ()
        cursor.execute(
            f"""
            UPDATE chat_sessions
            SET updated_at = ?, last_message_at = ?, last_message_preview = ?, last_agent = ?{title_update}
            WHERE session_id = ?
            """,
            (now, now, preview, current_agent or "") + title_args + (session_id,),
        )
    conn.commit()
    conn.close()
    return get_chat_message(message_id)


def _normalize_chat_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    for json_col in ("token_detail", "citations", "activated_skills", "tool_calls", "mcp_calls"):
        if msg.get(json_col):
            try:
                msg[json_col] = json.loads(msg[json_col])
            except Exception:
                pass
    msg["thinking_enabled"] = bool(msg.get("thinking_enabled"))
    return msg


def get_chat_message(message_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_messages WHERE id = ?", (message_id,))
    row = cursor.fetchone()
    conn.close()
    return _normalize_chat_message(dict(row)) if row else None


def get_previous_user_message(session_id: str, ai_message_id: int) -> Optional[Dict[str, Any]]:
    """Return the most recent user message before the given AI message in a session."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chat_messages WHERE session_id = ? AND role = 'user' AND id < ? ORDER BY id DESC LIMIT 1",
        (session_id, ai_message_id),
    )
    row = cursor.fetchone()
    conn.close()
    return _normalize_chat_message(dict(row)) if row else None


def list_chat_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    messages = []
    for r in rows:
        msg = _normalize_chat_message(dict(r))
        messages.append(msg)
    return messages


def delete_chat_messages(session_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# -----------------------------------------------------------------------------
# Chat Sessions & Human Handoff
# -----------------------------------------------------------------------------


def ensure_chat_session(session_id: str, title: Optional[str] = None) -> Dict[str, Any]:
    """Create a chat session row if it doesn't exist."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        now = now_cn()
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, handoff_status, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "none", title, now, now),
        )
        conn.commit()
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {"session_id": session_id, "handoff_status": "none"}


def create_chat_session(user_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Create a new chat session with a fresh session_id."""
    session_id = f"web-{uuid.uuid4().hex[:12]}"
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (session_id, handoff_status, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, "none", title, now, now),
    )
    conn.commit()
    conn.close()
    return get_chat_session(session_id) or {"session_id": session_id, "handoff_status": "none"}


def list_user_chat_sessions(user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent chat sessions ordered by last activity."""
    # user_id is accepted for future multi-tenant use; currently sessions are global per demo.
    _ = user_id
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM chat_sessions
        ORDER BY COALESCE(last_message_at, updated_at, created_at) DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _normalize_handoff_session(session: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not session:
        return None
    raw_package = session.get("handoff_package_json")
    if raw_package:
        try:
            session["handoff_package"] = json.loads(raw_package)
        except (TypeError, json.JSONDecodeError):
            session["handoff_package"] = None
    else:
        session["handoff_package"] = None
    return session


def _record_handoff_action(
    cursor: sqlite3.Cursor,
    session_id: str,
    action_type: str,
    status_before: Optional[str],
    status_after: Optional[str],
    actor: str,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    cursor.execute(
        """
        INSERT INTO handoff_actions
        (session_id, action_type, status_before, status_after, actor, action_detail, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            action_type,
            status_before,
            status_after,
            actor,
            json.dumps(detail or {}, ensure_ascii=False, default=str),
            now_cn(),
        ),
    )


def list_handoff_actions(session_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM handoff_actions WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = []
    for row in cursor.fetchall():
        item = dict(row)
        if item.get("action_detail"):
            try:
                item["action_detail"] = json.loads(item["action_detail"])
            except (TypeError, json.JSONDecodeError):
                pass
        rows.append(item)
    conn.close()
    return rows


def _load_session_for_transition(cursor: sqlite3.Cursor, session_id: str) -> Dict[str, Any]:
    cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError("会话不存在")
    return dict(row)


def request_handoff(
    session_id: str,
    reason: str,
    *,
    risk_level: str = "L3",
    reason_code: str = "owner_requested",
    queue: Optional[str] = "property_service",
    handoff_package: Optional[Dict[str, Any]] = None,
    actor: str = "owner",
) -> Dict[str, Any]:
    """Create or refresh a human-takeover request with its evidence package.

    A repeated request never silently takes a session away from an active staff
    member.  It only refreshes context and writes an audit action.
    """
    ensure_chat_session(session_id)
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    current = _load_session_for_transition(cursor, session_id)
    before = current.get("handoff_status") or "none"
    target = "requested" if before in {"none", "cancelled", "closed", "resolved"} else before
    package_json = json.dumps(handoff_package, ensure_ascii=False, default=str) if handoff_package else current.get("handoff_package_json")
    fields = [
        "handoff_reason = ?", "handoff_risk_level = ?", "handoff_reason_code = ?", "handoff_queue = ?",
        "handoff_package_json = ?", "handoff_last_actor = ?", "handoff_last_action_at = ?", "updated_at = ?",
    ]
    params: List[Any] = [reason, risk_level, reason_code, queue, package_json, actor, now, now]
    if target != before:
        fields.extend([
            "handoff_status = ?", "handoff_requested_at = ?", "handoff_active_at = NULL",
            "handoff_waiting_at = NULL", "handoff_resolved_at = NULL", "handoff_closed_at = NULL",
            "handoff_cancelled_at = NULL", "handoff_summary = NULL", "handoff_outcome = NULL", "assigned_to = NULL",
        ])
        params.extend([target, now])
    cursor.execute(f"UPDATE chat_sessions SET {', '.join(fields)} WHERE session_id = ?", tuple(params + [session_id]))
    _record_handoff_action(
        cursor, session_id, "request" if target != before else "request_refresh", before, target, actor,
        {"reason": reason, "risk_level": risk_level, "reason_code": reason_code, "queue": queue},
    )
    conn.commit()
    conn.close()
    return get_chat_session(session_id) or {"session_id": session_id, "handoff_status": target}


def _transition_handoff(
    session_id: str,
    target: str,
    *,
    action_type: str,
    actor: str,
    detail: Optional[Dict[str, Any]] = None,
    assigned_to: Optional[str] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_chat_session(session_id)
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    current = _load_session_for_transition(cursor, session_id)
    before = current.get("handoff_status") or "none"
    if not is_transition_allowed(before, target):
        conn.close()
        raise ValueError(f"人工协同状态不能从 {before} 变更为 {target}")
    fields = ["handoff_status = ?", "handoff_last_actor = ?", "handoff_last_action_at = ?", "updated_at = ?"]
    params: List[Any] = [target, actor, now, now]
    if target == "active":
        fields.extend(["assigned_to = ?", "handoff_active_at = ?"])
        params.extend([assigned_to or current.get("assigned_to") or actor, now])
    elif target == "waiting_user":
        fields.append("handoff_waiting_at = ?")
        params.append(now)
    elif target == "resolved":
        resolved_summary = summary or "工作人员已给出处理结论，等待业主确认。"
        fields.extend(["handoff_resolved_at = ?", "handoff_summary = ?", "handoff_outcome = ?"])
        params.extend([now, resolved_summary, resolved_summary])
    elif target == "closed":
        fields.append("handoff_closed_at = ?")
        params.append(now)
    elif target == "cancelled":
        fields.append("handoff_cancelled_at = ?")
        params.append(now)
    cursor.execute(f"UPDATE chat_sessions SET {', '.join(fields)} WHERE session_id = ?", tuple(params + [session_id]))
    _record_handoff_action(cursor, session_id, action_type, before, target, actor, detail)
    conn.commit()
    conn.close()
    return get_chat_session(session_id) or {"session_id": session_id, "handoff_status": target}


def claim_handoff(session_id: str, staff_name: str) -> Dict[str, Any]:
    return _transition_handoff(
        session_id, "active", action_type="claim", actor=staff_name, assigned_to=staff_name,
        detail={"assigned_to": staff_name},
    )


def activate_handoff(session_id: str, assigned_to: str) -> Dict[str, Any]:
    """Backward-compatible alias for the explicit staff claim action."""
    return claim_handoff(session_id, assigned_to)


def wait_for_handoff_user(session_id: str, staff_name: str, prompt: str) -> Dict[str, Any]:
    return _transition_handoff(
        session_id, "waiting_user", action_type="request_owner_input", actor=staff_name,
        detail={"prompt": prompt},
    )


def resume_handoff_after_owner_message(session_id: str) -> Dict[str, Any]:
    session = get_chat_session(session_id) or {}
    return _transition_handoff(
        session_id, "active", action_type="owner_supplied_requested_info", actor="owner",
        assigned_to=session.get("assigned_to") or "物业工作人员",
        detail={"note": "业主已补充工作人员要求的信息"},
    )


def resolve_handoff(session_id: str, resolution: Optional[str] = None, staff_name: str = "物业工作人员") -> Dict[str, Any]:
    return _transition_handoff(
        session_id, "resolved", action_type="resolve", actor=staff_name, summary=resolution,
        detail={"resolution": resolution or ""},
    )


def close_handoff(session_id: str, staff_name: str = "物业工作人员") -> Dict[str, Any]:
    return _transition_handoff(
        session_id, "closed", action_type="close", actor=staff_name,
        detail={"note": "人工协同闭环已关闭"},
    )


def cancel_handoff(session_id: str, actor: str = "owner", reason: Optional[str] = None) -> Dict[str, Any]:
    return _transition_handoff(
        session_id, "cancelled", action_type="cancel", actor=actor, detail={"reason": reason or ""},
    )


def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return _normalize_handoff_session(dict(row)) if row else None


def get_handoff_package(session_id: str) -> Dict[str, Any]:
    session = get_chat_session(session_id)
    if not session:
        raise ValueError("会话不存在")
    return {"session": session, "package": session.get("handoff_package") or {}, "actions": list_handoff_actions(session_id)}


def list_handoff_sessions(status: Optional[str] = None, include_completed: bool = False) -> List[Dict[str, Any]]:
    """Return actionable human-copilot sessions, including result review by default."""
    conn = _get_conn()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM chat_sessions WHERE handoff_status = ? ORDER BY handoff_last_action_at DESC, handoff_requested_at DESC", (status,))
    elif include_completed:
        cursor.execute("SELECT * FROM chat_sessions WHERE handoff_status != 'none' ORDER BY handoff_last_action_at DESC, handoff_requested_at DESC")
    else:
        cursor.execute("SELECT * FROM chat_sessions WHERE handoff_status IN ('requested', 'active', 'waiting_user', 'resolved') ORDER BY handoff_last_action_at DESC, handoff_requested_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_normalize_handoff_session(dict(row)) or {} for row in rows]


def is_handoff_active(session_id: str) -> bool:
    """Return True when AI must yield responsibility to a human."""
    session = get_chat_session(session_id)
    return session is not None and session.get("handoff_status") in {"requested", "active"}


def is_handoff_requested(session_id: str) -> bool:
    session = get_chat_session(session_id)
    return session is not None and session.get("handoff_status") == "requested"


# ---------------------------------------------------------------------------
# V1.3 Observability & Cost Governance
# ---------------------------------------------------------------------------


def create_chat_trace(
    trace_id: str,
    session_id: str,
    user_message: str,
    run_type: str = "chat",
    evaluation_case_id: Optional[int] = None,
    risk_level: Optional[str] = None,
    version_snapshot: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_traces
        (trace_id, session_id, user_message, status, run_type, evaluation_case_id, risk_level, version_snapshot, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trace_id, session_id, user_message, "in_progress", run_type, evaluation_case_id, risk_level, version_snapshot, now, now),
    )
    conn.commit()
    conn.close()
    return get_chat_trace(trace_id) or {"trace_id": trace_id, "session_id": session_id}


def update_chat_trace(
    trace_id: str,
    intent: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    run_type: Optional[str] = None,
    evaluation_case_id: Optional[int] = None,
    evaluation_run_id: Optional[int] = None,
    risk_level: Optional[str] = None,
    version_snapshot: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    fields = []
    params = []
    if intent is not None:
        fields.append("intent = ?")
        params.append(intent)
    if agent_name is not None:
        fields.append("agent_name = ?")
        params.append(agent_name)
    if agent_id is not None:
        fields.append("agent_id = ?")
        params.append(agent_id)
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if run_type is not None:
        fields.append("run_type = ?")
        params.append(run_type)
    if evaluation_case_id is not None:
        fields.append("evaluation_case_id = ?")
        params.append(evaluation_case_id)
    if evaluation_run_id is not None:
        fields.append("evaluation_run_id = ?")
        params.append(evaluation_run_id)
    if risk_level is not None:
        fields.append("risk_level = ?")
        params.append(risk_level)
    if version_snapshot is not None:
        fields.append("version_snapshot = ?")
        params.append(version_snapshot)
    if not fields:
        conn.close()
        return get_chat_trace(trace_id)
    fields.append("updated_at = ?")
    params.append(now)
    params.append(trace_id)
    cursor.execute(
        f"UPDATE chat_traces SET {', '.join(fields)} WHERE trace_id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_chat_trace(trace_id)


def get_chat_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_traces WHERE trace_id = ?", (trace_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_chat_traces(
    session_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    if session_id:
        cursor.execute(
            "SELECT * FROM chat_traces WHERE session_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
    else:
        cursor.execute(
            "SELECT * FROM chat_traces ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_model_call(
    trace_id: str,
    stage: str,
    model_id: str,
    model_selection_reason: Optional[str] = None,
    latency_ms: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    usage_source: str = "unavailable",
    status: str = "success",
    error_summary: Optional[str] = None,
    price_snapshot: Optional[Dict[str, Any]] = None,
    estimated_cost_cny: Optional[float] = None,
    context_breakdown: Optional[Dict[str, Any]] = None,
    usage_normalized: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO model_calls (
            trace_id, stage, model_id, status, latency_ms, input_tokens, output_tokens,
            reasoning_tokens, cached_tokens, total_tokens, usage_source, model_selection_reason,
            error_summary, price_snapshot, estimated_cost_cny, context_breakdown, usage_normalized, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            stage,
            model_id,
            status,
            latency_ms,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_tokens,
            total_tokens,
            usage_source,
            model_selection_reason,
            error_summary,
            json.dumps(price_snapshot, ensure_ascii=False) if price_snapshot else None,
            estimated_cost_cny,
            json.dumps(context_breakdown, ensure_ascii=False) if context_breakdown else None,
            json.dumps(usage_normalized, ensure_ascii=False) if usage_normalized else None,
            now,
        ),
    )
    call_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_model_call(call_id) or {"id": call_id}


def get_model_call(call_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_calls WHERE id = ?", (call_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    msg = dict(row)
    for json_col in ("price_snapshot", "context_breakdown"):
        if msg.get(json_col):
            try:
                msg[json_col] = json.loads(msg[json_col])
            except Exception:
                pass
    return msg


def get_model_calls_for_trace(trace_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM model_calls WHERE trace_id = ? ORDER BY id ASC",
        (trace_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    calls = []
    for r in rows:
        msg = dict(r)
        for json_col in ("price_snapshot", "context_breakdown"):
            if msg.get(json_col):
                try:
                    msg[json_col] = json.loads(msg[json_col])
                except Exception:
                    pass
        calls.append(msg)
    return calls


def record_mcp_call_audit(
    trace_id: str,
    server_name: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    status: str = "success",
    result_summary: Optional[str] = None,
    error_summary: Optional[str] = None,
    latency_ms: Optional[int] = None,
    invocation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO mcp_call_audits (
            trace_id, server_name, tool_name, arguments, status, result_summary,
            error_summary, latency_ms, invocation_mode, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            server_name,
            tool_name,
            json.dumps(arguments, ensure_ascii=False) if arguments else None,
            status,
            result_summary,
            error_summary,
            latency_ms,
            invocation_mode,
            now,
        ),
    )
    audit_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_mcp_call_audit(audit_id) or {"id": audit_id}


def get_mcp_call_audit(audit_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mcp_call_audits WHERE id = ?", (audit_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    msg = dict(row)
    if msg.get("arguments"):
        try:
            msg["arguments"] = json.loads(msg["arguments"])
        except Exception:
            pass
    return msg


def get_mcp_call_audits_for_trace(trace_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM mcp_call_audits WHERE trace_id = ? ORDER BY id ASC",
        (trace_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    audits = []
    for r in rows:
        msg = dict(r)
        if msg.get("arguments"):
            try:
                msg["arguments"] = json.loads(msg["arguments"])
            except Exception:
                pass
        audits.append(msg)
    return audits


def _normalize_price(row: Dict[str, Any]) -> Dict[str, Any]:
    row["enabled"] = bool(row.get("enabled"))
    return row


def create_model_price(
    model_id: str,
    effective_date: str,
    input_price_per_1m: Optional[float] = None,
    cached_input_price_per_1m: Optional[float] = None,
    output_price_per_1m: Optional[float] = None,
    reasoning_price_per_1m: Optional[float] = None,
    source_note: Optional[str] = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO model_prices (
            model_id, currency, effective_date, input_price_per_1m, cached_input_price_per_1m,
            output_price_per_1m, reasoning_price_per_1m, source_note, enabled, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            "CNY",
            effective_date,
            input_price_per_1m,
            cached_input_price_per_1m,
            output_price_per_1m,
            reasoning_price_per_1m,
            source_note,
            1 if enabled else 0,
            now,
            now,
        ),
    )
    price_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_model_price(price_id) or {"id": price_id}


def get_model_price(price_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_prices WHERE id = ?", (price_id,))
    row = cursor.fetchone()
    conn.close()
    return _normalize_price(dict(row)) if row else None


def list_model_prices(enabled_only: bool = False) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM model_prices WHERE 1=1"
    params = []
    if enabled_only:
        query += " AND enabled = 1"
    query += " ORDER BY model_id, effective_date DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_normalize_price(dict(r)) for r in rows]


def get_enabled_price_for_model(model_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM model_prices WHERE model_id = ? AND enabled = 1 ORDER BY effective_date DESC LIMIT 1",
        (model_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return _normalize_price(dict(row)) if row else None


def update_model_price(
    price_id: int,
    **updates: Any,
) -> Optional[Dict[str, Any]]:
    allowed = {
        "model_id",
        "effective_date",
        "input_price_per_1m",
        "cached_input_price_per_1m",
        "output_price_per_1m",
        "reasoning_price_per_1m",
        "source_note",
        "enabled",
    }
    fields = []
    params = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        if key == "enabled":
            value = 1 if value else 0
        fields.append(f"{key} = ?")
        params.append(value)
    if not fields:
        return get_model_price(price_id)
    fields.append("updated_at = ?")
    params.append(now_cn())
    params.append(price_id)
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE model_prices SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    return get_model_price(price_id)


def delete_model_price(price_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM model_prices WHERE id = ?", (price_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_budget_thresholds() -> Dict[str, Any]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM budget_thresholds WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {
            "id": 1,
            "per_call_threshold_cny": None,
            "daily_threshold_cny": None,
            "monthly_threshold_cny": None,
        }
    thresholds = dict(row)
    # Ensure the new key exists for clients using an older migrated row.
    thresholds.setdefault("monthly_threshold_cny", None)
    return thresholds


def update_budget_thresholds(
    per_call_threshold_cny: Optional[float] = None,
    daily_threshold_cny: Optional[float] = None,
    monthly_threshold_cny: Optional[float] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE budget_thresholds
        SET per_call_threshold_cny = ?, daily_threshold_cny = ?, monthly_threshold_cny = ?, updated_at = ?
        WHERE id = 1
        """,
        (per_call_threshold_cny, daily_threshold_cny, monthly_threshold_cny, now),
    )
    if cursor.rowcount == 0:
        cursor.execute(
            "INSERT OR IGNORE INTO budget_thresholds (id, per_call_threshold_cny, daily_threshold_cny, monthly_threshold_cny, updated_at) VALUES (1, ?, ?, ?, ?)",
            (per_call_threshold_cny, daily_threshold_cny, monthly_threshold_cny, now),
        )
    conn.commit()
    conn.close()
    return get_budget_thresholds()


# ---------------------------------------------------------------------------
# Agents CRUD
# ---------------------------------------------------------------------------


def create_agent(
    agent_id: str,
    name: str,
    description: str = "",
    instructions: str = "",
    category: str = "vertical",
    enabled: bool = True,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO agents
        (agent_id, name, description, instructions, category, enabled, model_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (agent_id, name, description, instructions, category, 1 if enabled else 0, model_id, now, now),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_agent(row_id)


def get_agent(agent_row_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agents WHERE id = ?", (agent_row_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_with_bool(row, "enabled") if row else None


def get_agent_by_agent_id(agent_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_with_bool(row, "enabled") if row else None


def list_agents(category: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM agents WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY id"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_with_bool(r, "enabled") for r in rows]


def update_agent(
    agent_row_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    category: Optional[str] = None,
    enabled: Optional[bool] = None,
    model_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [
        ("name", name),
        ("description", description),
        ("instructions", instructions),
        ("category", category),
        ("model_id", model_id),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    if enabled is not None:
        fields.append("enabled = ?")
        params.append(1 if enabled else 0)
    params.append(agent_row_id)
    cursor.execute(
        f"UPDATE agents SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_agent(agent_row_id)


def delete_agent(agent_row_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM agents WHERE id = ?", (agent_row_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def set_agent_skills(agent_id: str, skill_ids: List[int]):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM agent_skills WHERE agent_id = ?", (agent_id,))
    for skill_id in skill_ids:
        cursor.execute(
            "INSERT INTO agent_skills (agent_id, skill_id) VALUES (?, ?)",
            (agent_id, skill_id),
        )
    conn.commit()
    conn.close()


def get_agent_skills(agent_id: str) -> List[int]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT skill_id FROM agent_skills WHERE agent_id = ?", (agent_id,))
    rows = cursor.fetchall()
    conn.close()
    return [r["skill_id"] for r in rows]


def set_agent_tools(agent_id: str, tools: List[Dict[str, Any]]):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM agent_tools WHERE agent_id = ?", (agent_id,))
    for tool in tools:
        cursor.execute(
            "INSERT INTO agent_tools (agent_id, tool_name, config) VALUES (?, ?, ?)",
            (agent_id, tool["tool_name"], json.dumps(tool.get("config"), ensure_ascii=False) if tool.get("config") else None),
        )
    conn.commit()
    conn.close()


def get_agent_tools(agent_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agent_tools WHERE agent_id = ?", (agent_id,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("config"):
            try:
                d["config"] = json.loads(d["config"])
            except Exception:
                pass
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Retrieval Settings CRUD
# ---------------------------------------------------------------------------


def get_retrieval_settings(name: str = "default") -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM retrieval_settings WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    r = _row_with_bool(row, "enable_rerank")
    return r


def update_retrieval_settings(
    name: str = "default",
    top_k: Optional[int] = None,
    keyword_weight: Optional[float] = None,
    semantic_weight: Optional[float] = None,
    rrf_k: Optional[int] = None,
    enable_rerank: Optional[bool] = None,
    rerank_model: Optional[str] = None,
    score_threshold: Optional[float] = None,
    context_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    existing = get_retrieval_settings(name)
    if not existing:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retrieval_settings
            (name, top_k, keyword_weight, semantic_weight, rrf_k, enable_rerank, rerank_model, score_threshold, context_threshold, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                top_k or 5,
                keyword_weight if keyword_weight is not None else 0.3,
                semantic_weight if semantic_weight is not None else 0.7,
                rrf_k or 60,
                1 if enable_rerank else 0,
                rerank_model,
                score_threshold if score_threshold is not None else 0.0,
                context_threshold if context_threshold is not None else 0.2,
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return get_retrieval_settings(name)

    conn = _get_conn()
    cursor = conn.cursor()
    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [
        ("top_k", top_k),
        ("keyword_weight", keyword_weight),
        ("semantic_weight", semantic_weight),
        ("rrf_k", rrf_k),
        ("rerank_model", rerank_model),
        ("score_threshold", score_threshold),
        ("context_threshold", context_threshold),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    if enable_rerank is not None:
        fields.append("enable_rerank = ?")
        params.append(1 if enable_rerank else 0)
    params.append(name)
    cursor.execute(
        f"UPDATE retrieval_settings SET {', '.join(fields)} WHERE name = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_retrieval_settings(name)


# ---------------------------------------------------------------------------
# Model Configs CRUD
# ---------------------------------------------------------------------------


def create_model_config(
    model_id: str,
    name: str,
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    is_default: bool = False,
    enabled: bool = True,
    description: str = "",
) -> Dict[str, Any]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    if is_default:
        cursor.execute("UPDATE model_configs SET is_default = 0")
    cursor.execute(
        """
        INSERT INTO model_configs
        (model_id, name, provider, api_key, base_url, model_params, is_default, enabled, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            name,
            provider,
            api_key,
            base_url,
            json.dumps(model_params, ensure_ascii=False) if model_params else None,
            1 if is_default else 0,
            1 if enabled else 0,
            description,
            now,
            now,
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_model_config(row_id)


def get_model_config(config_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_configs WHERE id = ?", (config_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return _parse_model_config(row)


def get_model_config_by_model_id(model_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_configs WHERE model_id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return _parse_model_config(row)


def get_default_model_config() -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_configs WHERE is_default = 1 AND enabled = 1 LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if not row:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_configs WHERE enabled = 1 ORDER BY id LIMIT 1")
        row = cursor.fetchone()
        conn.close()
    if not row:
        return None
    return _parse_model_config(row)


def _parse_model_config(row: sqlite3.Row) -> Dict[str, Any]:
    d = _row_with_bool(row, "enabled")
    d = _row_with_bool(d, "is_default")
    if d.get("model_params"):
        try:
            d["model_params"] = json.loads(d["model_params"])
        except Exception:
            d["model_params"] = {}
    else:
        d["model_params"] = {}
    return d


def list_model_configs() -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_configs ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [_parse_model_config(r) for r in rows]


def update_model_config(
    config_id: int,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    is_default: Optional[bool] = None,
    enabled: Optional[bool] = None,
    description: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    if is_default:
        cursor.execute("UPDATE model_configs SET is_default = 0")
    fields = ["updated_at = ?"]
    params = [now]
    for col, val in [
        ("name", name),
        ("provider", provider),
        ("api_key", api_key),
        ("base_url", base_url),
        ("description", description),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    if model_params is not None:
        fields.append("model_params = ?")
        params.append(json.dumps(model_params, ensure_ascii=False))
    if is_default is not None:
        fields.append("is_default = ?")
        params.append(1 if is_default else 0)
    if enabled is not None:
        fields.append("enabled = ?")
        params.append(1 if enabled else 0)
    params.append(config_id)
    cursor.execute(
        f"UPDATE model_configs SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()
    return get_model_config(config_id)


def set_default_model_config(config_id: int) -> Optional[Dict[str, Any]]:
    return update_model_config(config_id, is_default=True)


def delete_model_config(config_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM model_configs WHERE id = ?", (config_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ---------------------------------------------------------------------------
# MCP Tools CRUD & Discovery
# ---------------------------------------------------------------------------


def save_mcp_tool(
    server_id: int,
    name: str,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    tool_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO mcp_tools (server_id, name, description, input_schema, tool_metadata)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(server_id, name) DO UPDATE SET
            description = excluded.description,
            input_schema = excluded.input_schema,
            tool_metadata = excluded.tool_metadata
        """,
        (
            server_id,
            name,
            description,
            json.dumps(input_schema, ensure_ascii=False) if input_schema else "{}",
            json.dumps(tool_metadata, ensure_ascii=False) if tool_metadata else None,
        ),
    )
    cursor.execute(
        "SELECT id FROM mcp_tools WHERE server_id = ? AND name = ?",
        (server_id, name),
    )
    selected = cursor.fetchone()
    tool_id = int(selected["id"]) if selected else int(cursor.lastrowid or 0)
    conn.commit()
    conn.close()
    return get_mcp_tool(tool_id) or {}


def get_mcp_tool(tool_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mcp_tools WHERE id = ?", (tool_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return _parse_mcp_tool(row)


def _parse_mcp_tool(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for col in ("input_schema", "tool_metadata"):
        if d.get(col):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                pass
    return d


def list_mcp_tools(server_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    if server_id is not None:
        cursor.execute("SELECT * FROM mcp_tools WHERE server_id = ? ORDER BY id", (server_id,))
    else:
        cursor.execute("SELECT * FROM mcp_tools ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [_parse_mcp_tool(r) for r in rows]


def delete_mcp_tools_for_server(server_id: int):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM mcp_tools WHERE server_id = ?", (server_id,))
    conn.commit()
    conn.close()


def toggle_mcp_server_enabled(server_id: int, enabled: bool) -> Optional[Dict[str, Any]]:
    now = now_cn("%Y-%m-%d %H:%M")
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE mcp_servers SET enabled = ?, updated_at = ? WHERE id = ?",
        (1 if enabled else 0, now, server_id),
    )
    conn.commit()
    conn.close()
    return get_mcp_server(server_id)


# ---------------------------------------------------------------------------
# V1.8 runtime control plane
# ---------------------------------------------------------------------------

def _parse_runtime_release(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["config"] = _parse_json_text(item.pop("config_json", None), {})
    item["validation"] = _parse_json_text(item.pop("validation_json", None), {})
    return item


def next_runtime_release_version() -> int:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM runtime_releases")
    value = int(cursor.fetchone()[0])
    conn.close()
    return value


def create_runtime_release(
    release_id: str,
    version: int,
    config_hash: str,
    config: Dict[str, Any],
    validation: Dict[str, Any],
    parent_release_id: Optional[str] = None,
    created_by: str = "system",
    status: str = "draft",
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO runtime_releases (
            release_id, version, status, config_hash, config_json,
            validation_json, parent_release_id, created_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            release_id,
            int(version),
            status,
            config_hash,
            _json_text(config),
            _json_text(validation),
            parent_release_id,
            created_by,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return get_runtime_release(release_id) or {}


def get_runtime_release(release_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runtime_releases WHERE release_id = ?", (release_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_runtime_release(row)


def get_current_runtime_release() -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT r.*
        FROM runtime_release_pointer p
        JOIN runtime_releases r ON r.release_id = p.release_id
        WHERE p.pointer_key = 'current'
        """
    )
    row = cursor.fetchone()
    conn.close()
    return _parse_runtime_release(row)


def list_runtime_releases(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM runtime_releases ORDER BY version DESC LIMIT ?",
        (max(1, min(int(limit), 200)),),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_parse_runtime_release(row) or {} for row in rows]


def publish_runtime_release(release_id: str) -> Dict[str, Any]:
    """Atomically move the current pointer to one validated draft release."""
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            "SELECT status, validation_json FROM runtime_releases WHERE release_id = ?",
            (release_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError("runtime release not found")
        validation = _parse_json_text(row["validation_json"], {})
        if not validation.get("valid"):
            raise ValueError("runtime release validation failed")

        cursor.execute(
            """
            SELECT release_id FROM runtime_release_pointer
            WHERE pointer_key = 'current'
            """
        )
        current = cursor.fetchone()
        if current and current["release_id"] != release_id:
            cursor.execute(
                """
                UPDATE runtime_releases
                SET status = 'superseded', superseded_at = ?
                WHERE release_id = ? AND status = 'published'
                """,
                (now, current["release_id"]),
            )
        cursor.execute(
            """
            UPDATE runtime_releases
            SET status = 'published', published_at = ?, superseded_at = NULL
            WHERE release_id = ?
            """,
            (now, release_id),
        )
        cursor.execute(
            """
            INSERT INTO runtime_release_pointer (pointer_key, release_id, updated_at)
            VALUES ('current', ?, ?)
            ON CONFLICT(pointer_key) DO UPDATE SET
                release_id = excluded.release_id,
                updated_at = excluded.updated_at
            """,
            (release_id, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_runtime_release(release_id) or {}


def rollback_runtime_release(release_id: str) -> Dict[str, Any]:
    """Re-publish a historical validated snapshot without recompiling it."""
    return publish_runtime_release(release_id)


def save_run_config_snapshot(
    snapshot_id: str,
    session_id: str,
    release_id: str,
    config_hash: str,
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Pin the first release seen by a session; never overwrite it."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO run_config_snapshots (
            snapshot_id, session_id, release_id, config_hash, snapshot_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (snapshot_id, session_id, release_id, config_hash, _json_text(snapshot), now_cn()),
    )
    conn.commit()
    conn.close()
    return get_run_config_snapshot(session_id) or {}


def get_run_config_snapshot(session_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM run_config_snapshots WHERE session_id = ?",
        (session_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    item["snapshot"] = _parse_json_text(item.pop("snapshot_json", None), {})
    return item


def replace_tool_policies(release_id: str, policies: List[Dict[str, Any]]) -> None:
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute("DELETE FROM tool_policies WHERE release_id = ?", (release_id,))
        for policy in policies:
            cursor.execute(
                """
                INSERT INTO tool_policies (
                    release_id, server_id, server_name, tool_name, effect,
                    risk_level, allowed_paths_json, requires_confirmation,
                    enabled, policy_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    release_id,
                    policy.get("server_id"),
                    policy.get("server_name") or "",
                    policy.get("tool_name") or "",
                    policy.get("effect") or "unknown",
                    policy.get("risk_level") or "L3",
                    _json_text(policy.get("allowed_paths") or []),
                    1 if policy.get("requires_confirmation") else 0,
                    1 if policy.get("enabled", True) else 0,
                    policy.get("policy_reason"),
                    now_cn(),
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_tool_policies(release_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM tool_policies WHERE release_id = ? ORDER BY server_name, tool_name",
        (release_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        item = dict(row)
        item["allowed_paths"] = _parse_json_text(item.pop("allowed_paths_json", None), [])
        item["requires_confirmation"] = bool(item.get("requires_confirmation"))
        item["enabled"] = bool(item.get("enabled"))
        result.append(item)
    return result


def save_evidence_ledger(
    trace_id: str,
    session_id: str,
    ledger: Dict[str, Any],
    release_id: Optional[str] = None,
    config_hash: Optional[str] = None,
    runtime_path: Optional[str] = None,
    status: str = "running",
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO run_evidence_ledgers (
            trace_id, session_id, release_id, config_hash, runtime_path,
            status, ledger_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trace_id) DO UPDATE SET
            release_id = excluded.release_id,
            config_hash = excluded.config_hash,
            runtime_path = excluded.runtime_path,
            status = excluded.status,
            ledger_json = excluded.ledger_json,
            updated_at = excluded.updated_at
        """,
        (
            trace_id,
            session_id,
            release_id,
            config_hash,
            runtime_path,
            status,
            _json_text(ledger),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return get_evidence_ledger(trace_id) or {}


def set_agent_knowledge_bindings(agent_id: str, knowledge_doc_ids: List[int]) -> None:
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            """
            INSERT INTO agent_knowledge_scopes (agent_id, binding_mode, updated_at)
            VALUES (?, 'explicit', ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                binding_mode = 'explicit',
                updated_at = excluded.updated_at
            """,
            (agent_id, now_cn()),
        )
        cursor.execute(
            "DELETE FROM agent_knowledge_bindings WHERE agent_id = ?",
            (agent_id,),
        )
        for doc_id in sorted({int(item) for item in knowledge_doc_ids}):
            cursor.execute(
                """
                INSERT INTO agent_knowledge_bindings (
                    agent_id, knowledge_doc_id, enabled
                ) VALUES (?, ?, 1)
                """,
                (agent_id, doc_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_agent_knowledge_bindings(agent_id: str) -> Optional[List[int]]:
    """Return None for legacy compatibility scope, or an explicit ID list."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT binding_mode FROM agent_knowledge_scopes WHERE agent_id = ?",
        (agent_id,),
    )
    scope = cursor.fetchone()
    if not scope:
        conn.close()
        return None
    cursor.execute(
        """
        SELECT knowledge_doc_id FROM agent_knowledge_bindings
        WHERE agent_id = ? AND enabled = 1
        ORDER BY knowledge_doc_id
        """,
        (agent_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [int(row["knowledge_doc_id"]) for row in rows]


def get_evidence_ledger(trace_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM run_evidence_ledgers WHERE trace_id = ?", (trace_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    item["ledger"] = _parse_json_text(item.pop("ledger_json", None), {})
    return item


def create_action_proposal(
    proposal_id: str,
    session_id: str,
    action_type: str,
    risk_level: str,
    payload: Dict[str, Any],
    idempotency_key: str,
    trace_id: Optional[str] = None,
    release_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO action_proposals (
            proposal_id, session_id, trace_id, release_id, action_type,
            risk_level, payload_json, status, idempotency_key, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending_confirmation', ?, ?, ?)
        """,
        (
            proposal_id,
            session_id,
            trace_id,
            release_id,
            action_type,
            risk_level,
            _json_text(payload),
            idempotency_key,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return get_action_proposal_by_idempotency_key(idempotency_key) or {}


def _parse_action_proposal(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["payload"] = _parse_json_text(item.pop("payload_json", None), {})
    return item


def get_action_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM action_proposals WHERE proposal_id = ?", (proposal_id,))
    row = cursor.fetchone()
    conn.close()
    return _parse_action_proposal(row)


def get_action_proposal_by_idempotency_key(idempotency_key: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM action_proposals WHERE idempotency_key = ?",
        (idempotency_key,),
    )
    row = cursor.fetchone()
    conn.close()
    return _parse_action_proposal(row)


def get_pending_action_proposal(session_id: str, action_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    if action_type:
        cursor.execute(
            """
            SELECT * FROM action_proposals
            WHERE session_id = ? AND action_type = ? AND status = 'pending_confirmation'
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, action_type),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM action_proposals
            WHERE session_id = ? AND status = 'pending_confirmation'
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id,),
        )
    row = cursor.fetchone()
    conn.close()
    return _parse_action_proposal(row)


def get_latest_action_proposal(session_id: str, action_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    if action_type:
        cursor.execute(
            """
            SELECT * FROM action_proposals
            WHERE session_id = ? AND action_type = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, action_type),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM action_proposals
            WHERE session_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id,),
        )
    row = cursor.fetchone()
    conn.close()
    return _parse_action_proposal(row)


def record_action_approval(
    proposal_id: str,
    decision: str,
    actor: str,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            "SELECT status FROM action_proposals WHERE proposal_id = ?",
            (proposal_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError("action proposal not found")
        if row["status"] not in {"pending_confirmation", decision}:
            raise ValueError("action proposal is not pending confirmation")
        cursor.execute(
            """
            INSERT INTO action_approvals (proposal_id, decision, actor, comment, decided_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (proposal_id, decision, actor, comment, now),
        )
        cursor.execute(
            "UPDATE action_proposals SET status = ?, updated_at = ? WHERE proposal_id = ?",
            (decision, now, proposal_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_action_proposal(proposal_id) or {}


def list_action_approvals(proposal_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT proposal_id, decision, actor, comment, decided_at
        FROM action_approvals
        WHERE proposal_id = ?
        ORDER BY id ASC
        """,
        (proposal_id,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_action_receipt_by_idempotency_key(idempotency_key: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM action_receipts WHERE idempotency_key = ?",
        (idempotency_key,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    item["result"] = _parse_json_text(item.pop("result_json", None), {})
    return item


def save_runtime_acceptance_run(
    acceptance_run_id: str,
    case_key: str,
    release_id: Optional[str],
    status: str,
    evidence: Dict[str, Any],
    cleanup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO runtime_acceptance_runs (
            acceptance_run_id, case_key, release_id, status,
            evidence_json, cleanup_json, created_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(acceptance_run_id) DO UPDATE SET
            status = excluded.status,
            evidence_json = excluded.evidence_json,
            cleanup_json = excluded.cleanup_json,
            completed_at = excluded.completed_at
        """,
        (
            acceptance_run_id,
            case_key,
            release_id,
            status,
            _json_text(evidence),
            _json_text(cleanup or {}),
            now,
            now if status in {"passed", "failed"} else None,
        ),
    )
    conn.commit()
    cursor.execute(
        "SELECT * FROM runtime_acceptance_runs WHERE acceptance_run_id = ?",
        (acceptance_run_id,),
    )
    row = cursor.fetchone()
    conn.close()
    item = dict(row) if row else {}
    item["evidence"] = _parse_json_text(item.pop("evidence_json", None), {})
    item["cleanup"] = _parse_json_text(item.pop("cleanup_json", None), {})
    return item


def list_runtime_acceptance_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM runtime_acceptance_runs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (max(1, min(int(limit), 200)),),
    )
    rows = cursor.fetchall()
    conn.close()
    result: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["evidence"] = _parse_json_text(item.pop("evidence_json", None), {})
        item["cleanup"] = _parse_json_text(item.pop("cleanup_json", None), {})
        result.append(item)
    return result


def save_action_receipt(
    receipt_id: str,
    proposal_id: str,
    idempotency_key: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    error_summary: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_cn()
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            """
            INSERT OR IGNORE INTO action_receipts (
                receipt_id, proposal_id, idempotency_key, status,
                resource_type, resource_id, result_json, committed_at,
                error_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                proposal_id,
                idempotency_key,
                status,
                resource_type,
                resource_id,
                _json_text(result or {}),
                now if status == "committed" else None,
                error_summary,
                now,
            ),
        )
        cursor.execute(
            "UPDATE action_proposals SET status = ?, updated_at = ? WHERE proposal_id = ?",
            ("committed" if status == "committed" else "failed", now, proposal_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_action_receipt_by_idempotency_key(idempotency_key) or {}
