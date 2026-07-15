"""
Badcase Closed-Loop API
=======================

Implements the full badcase lifecycle:
    pending -> classified -> fixing -> verifying -> closed/rejected

Supports automatic classification, knowledge extraction, Darwin skill
optimization, model switch retry, and verification.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.settings import MODEL, build_model
from db.property_db import (
    add_badcase_action,
    create_badcase as db_create_badcase,
    create_capability_gap_draft as db_create_capability_gap_draft,
    create_knowledge_doc as db_create_knowledge_doc,
    create_knowledge_draft as db_create_knowledge_draft,
    create_skill as db_create_skill,
    create_skill_prompt_draft as db_create_skill_prompt_draft,
    delete_badcase as db_delete_badcase,
    get_badcase as db_get_badcase,
    get_capability_gap_draft as db_get_capability_gap_draft,
    get_chat_message,
    get_enabled_price_for_model,
    get_knowledge_draft as db_get_knowledge_draft,
    get_skill,
    get_skill_by_name,
    get_skill_prompt_draft as db_get_skill_prompt_draft,
    list_badcase_actions,
    list_badcases as db_list_badcases,
    list_capability_gap_drafts as db_list_capability_gap_drafts,
    list_knowledge_drafts as db_list_knowledge_drafts,
    list_skill_prompt_drafts as db_list_skill_prompt_drafts,
    list_skills,
    record_model_call,
    update_badcase as db_update_badcase,
    update_capability_gap_draft as db_update_capability_gap_draft,
    update_knowledge_draft as db_update_knowledge_draft,
    update_skill as db_update_skill,
    update_skill_prompt_draft as db_update_skill_prompt_draft,
)

router = APIRouter(tags=["badcases"])

VALID_CATEGORIES = {
    "knowledge_gap",   # 知识库缺口
    "skill_prompt",    # Skill/Prompt 问题
    "mcp_capability",  # MCP/能力缺口
    "routing",         # 路由问题
    "response_quality", # 模型回答质量
    "other",
    "pending",
}
VALID_STATUSES = {"pending", "classified", "fixing", "verifying", "closed", "rejected"}

CATEGORY_LABELS = {
    "knowledge_gap": "知识库缺口",
    "skill_prompt": "Skill/Prompt 问题",
    "mcp_capability": "MCP/能力缺口",
    "routing": "路由问题",
    "response_quality": "模型回答质量",
    "other": "其他",
    "pending": "待分类",
}


def _enrich_badcase(badcase: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add frontend-compatible aliases to a badcase record."""
    if not badcase:
        return None
    enriched = dict(badcase)
    # Preserve new fields if present; only alias legacy records.
    if not enriched.get("query"):
        enriched["query"] = enriched.get("original_query") or enriched.get("title") or "-"
    if not enriched.get("feedback_reason"):
        enriched["feedback_reason"] = enriched.get("rejected_reason") or "-"
    if not enriched.get("analysis_category"):
        enriched["analysis_category"] = enriched.get("category") or "-"
    enriched["category_label"] = CATEGORY_LABELS.get(enriched.get("category", ""), enriched.get("category", "-"))
    enriched["source_label"] = "人工反馈" if enriched.get("source") == "manual" else (enriched.get("source") or "auto")
    # Parse context_json for frontend convenience.
    context_json = enriched.get("context_json")
    if isinstance(context_json, str) and context_json:
        try:
            enriched["context"] = json.loads(context_json)
        except Exception:
            enriched["context"] = {}
    else:
        enriched["context"] = context_json or {}
    retest_context_json = enriched.get("retest_context_json")
    if isinstance(retest_context_json, str) and retest_context_json:
        try:
            enriched["retest_context"] = json.loads(retest_context_json)
        except Exception:
            enriched["retest_context"] = {}
    else:
        enriched["retest_context"] = retest_context_json or {}
    # Fallback to description so legacy/demo records still show a useful summary.
    enriched["analysis_evidence"] = (
        enriched.get("evidence")
        or enriched.get("root_cause")
        or enriched.get("description")
        or "暂无分析"
    )
    return enriched


class BadcaseCreate(BaseModel):
    title: str
    description: str = ""
    category: str = "other"
    status: str = "pending"
    evidence: str = ""
    source_message_id: Optional[int] = None
    session_id: Optional[str] = None
    source: str = "manual"
    original_query: Optional[str] = None
    ai_response: Optional[str] = None
    feedback_reason: Optional[str] = None
    priority: str = "medium"


class BadcaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None
    root_cause: Optional[str] = None
    fix_plan: Optional[str] = None
    rejected_reason: Optional[str] = None
    priority: Optional[str] = None


class ClassifyRequest(BaseModel):
    auto: bool = True
    category: Optional[str] = None
    reason: str = ""


class ExtractKnowledgeRequest(BaseModel):
    auto: bool = True
    title: Optional[str] = None
    content: Optional[str] = None
    category: str = "未分类"


class DarwinFixRequest(BaseModel):
    prompt: Optional[str] = None


class SwitchModelRetryRequest(BaseModel):
    model_id: Optional[str] = None
    user_message: Optional[str] = None


class VerifyRequest(BaseModel):
    passed: bool = True
    note: str = ""


class RejectRequest(BaseModel):
    rejected_reason: str = ""


class TransitionRequest(BaseModel):
    status: str = "verifying"
    note: str = ""


class PublishSkillDraftRequest(BaseModel):
    target_skill_id: Optional[int] = None


class AcceptGapRequest(BaseModel):
    note: str = ""


def _record_action(badcase_id: int, action_type: str, detail: Any, before: str, after: str, created_by: str = "system"):
    """Record a badcase lifecycle action."""
    return add_badcase_action(
        badcase_id=badcase_id,
        action_type=action_type,
        action_detail=json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail,
        status_before=before,
        status_after=after,
        created_by=created_by,
    )


def _extract_usage(usage_obj: Any) -> Dict[str, Optional[int]]:
    if isinstance(usage_obj, dict):
        return {
            "input_tokens": usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens"),
            "output_tokens": usage_obj.get("output_tokens") or usage_obj.get("completion_tokens"),
            "reasoning_tokens": usage_obj.get("reasoning_tokens"),
            "cached_tokens": usage_obj.get("cached_tokens") or usage_obj.get("prompt_cache_hit_tokens"),
            "total_tokens": usage_obj.get("total_tokens"),
        }
    return {
        "input_tokens": getattr(usage_obj, "input_tokens", None) or getattr(usage_obj, "prompt_tokens", None),
        "output_tokens": getattr(usage_obj, "output_tokens", None) or getattr(usage_obj, "completion_tokens", None),
        "reasoning_tokens": getattr(usage_obj, "reasoning_tokens", None),
        "cached_tokens": getattr(usage_obj, "cached_tokens", None) or getattr(usage_obj, "prompt_cache_hit_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
    }


async def _collect_response(generator) -> Tuple[str, Dict[str, Optional[int]]]:
    """Collect text and usage from an Agno async generator or a single response."""
    response = ""
    usage = {}
    try:
        if isinstance(generator, str):
            return generator, usage
        if hasattr(generator, "__aiter__"):
            async for chunk in generator:
                if hasattr(chunk, "content") and chunk.content:
                    response += str(chunk.content)
                elif hasattr(chunk, "delta") and chunk.delta:
                    response += str(chunk.delta)
                elif isinstance(chunk, str):
                    response += chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = _extract_usage(chunk.usage)
            return response.strip(), usage
        result = await generator
        if hasattr(result, "content"):
            if hasattr(result, "usage") and result.usage:
                usage = _extract_usage(result.usage)
            return str(result.content).strip(), usage
        if isinstance(result, str):
            return result.strip(), usage
        return "", usage
    except Exception:
        import traceback
        traceback.print_exc()
        return "", usage


async def _llm_generate(prompt: str, model: Optional[Any] = None, model_id: Optional[str] = None) -> Tuple[str, Dict[str, Optional[int]]]:
    """Generate text using the default, a provided model, or a model_id."""
    from agno.agent import Agent

    selected_model = model
    if model_id:
        selected_model = build_model(model_id)
    agent = Agent(model=selected_model or MODEL, markdown=False)
    return await _collect_response(agent.arun(prompt, stream=False))


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a text block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _find_darwin_skill() -> Optional[Dict[str, Any]]:
    """Find the Darwin optimization skill by name."""
    for name in ("达尔文", "darwin", "Darwin"):
        skill = get_skill_by_name(name)
        if skill:
            return skill
    return None


@router.get("")
async def list_badcases(status: Optional[str] = None, category: Optional[str] = None):
    """List badcases with optional filters."""
    if status and status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status: {status}")
    if category and category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"invalid category: {category}")
    cases = db_list_badcases(status=status, category=category)
    return {"badcases": [_enrich_badcase(c) for c in cases], "count": len(cases)}


@router.get("/{case_id}")
async def get_badcase(case_id: int):
    """Get a single badcase with actions and drafts."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    case["actions"] = list_badcase_actions(case_id)
    case["knowledge_drafts"] = [d for d in db_list_knowledge_drafts() if d.get("badcase_id") == case_id]
    case["skill_prompt_drafts"] = db_list_skill_prompt_drafts(badcase_id=case_id)
    case["capability_gap_drafts"] = db_list_capability_gap_drafts(badcase_id=case_id)
    return {"badcase": _enrich_badcase(case)}


@router.post("")
async def create_badcase(request: BadcaseCreate):
    """Create a new badcase."""
    if request.category not in VALID_CATEGORIES:
        request.category = "other"
    if request.status not in VALID_STATUSES:
        request.status = "pending"
    case = db_create_badcase(
        title=request.title,
        description=request.description,
        category=request.category,
        status=request.status,
        evidence=request.evidence,
        source_message_id=request.source_message_id,
        session_id=request.session_id,
        source=request.source,
        original_query=request.original_query,
        ai_response=request.ai_response,
        feedback_reason=request.feedback_reason,
        priority=request.priority,
    )
    return {"badcase": _enrich_badcase(case)}


@router.put("/{case_id}")
async def update_badcase(case_id: int, request: BadcaseUpdate):
    """Update badcase fields."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    updated = db_update_badcase(
        case_id=case_id,
        title=request.title,
        description=request.description,
        category=request.category,
        status=request.status,
        evidence=request.evidence,
        root_cause=request.root_cause,
        fix_plan=request.fix_plan,
        rejected_reason=request.rejected_reason,
        priority=request.priority,
    )
    if updated:
        _record_action(
            case_id, "update", request.dict(exclude_unset=True),
            case["status"], updated["status"], "user"
        )
    return {"badcase": _enrich_badcase(updated)}


@router.delete("/{case_id}")
async def delete_badcase(case_id: int):
    """Delete a badcase."""
    deleted = db_delete_badcase(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "deleted_id": case_id}


@router.post("/{case_id}/classify")
async def classify_badcase(case_id: int, request: ClassifyRequest = ClassifyRequest()):
    """Classify a badcase into one of the operational categories."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if case["status"] not in ("pending", "classified"):
        raise HTTPException(status_code=400, detail=f"cannot classify from status {case['status']}")

    context = case.get("context_json") or ""
    if isinstance(context, str) and context:
        try:
            context_obj = json.loads(context)
        except Exception:
            context_obj = {}
    else:
        context_obj = context or {}

    if request.auto:
        prompt = (
            "你是一名 AI 运营问题分类专家。请根据下面的 Badcase 信息，从以下类别中选择一个最贴切的，并给出理由和优先级：\n"
            "- knowledge_gap：知识库内容缺失、错误或未命中\n"
            "- skill_prompt：Skill 触发条件或 Prompt 指令缺陷\n"
            "- mcp_capability：MCP/外部工具/系统能力缺失或调用失败\n"
            "- routing：意图路由错误\n"
            "- response_quality：模型回复质量差、格式错误、未遵循指令\n"
            "- other：其他\n\n"
            f"标题：{case['title']}\n"
            f"描述：{case.get('description', '')}\n"
            f"反馈原因：{case.get('feedback_reason', '')}\n"
            f"原问题：{case.get('original_query', '')}\n"
            f"原回答：{case.get('ai_response', '')[:500]}\n"
            f"上下文：{json.dumps(context_obj, ensure_ascii=False)[:800]}\n\n"
            "请严格输出 JSON：{\"category\": \"<类别>\", \"reason\": \"<一句话理由>\", \"priority\": \"<high|medium|low>\"}"
        )
        raw, _ = await _llm_generate(prompt)
        parsed = _extract_json(raw) or {}
        category = parsed.get("category", "other")
        reason = parsed.get("reason", "自动分类失败，归入 other")
        priority = parsed.get("priority", "medium")
        if category not in VALID_CATEGORIES:
            category = "other"
        if priority not in ("high", "medium", "low"):
            priority = "medium"
    else:
        category = request.category or "other"
        reason = request.reason
        priority = "medium"
        if category not in VALID_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"invalid category: {category}")

    new_status = "classified"
    updated = db_update_badcase(
        case_id,
        category=category,
        status=new_status,
        root_cause=reason,
        priority=priority,
    )
    _record_action(
        case_id,
        "classify",
        {"category": category, "reason": reason, "priority": priority, "raw": raw if request.auto else None},
        case["status"],
        new_status,
    )
    return {"badcase": _enrich_badcase(updated), "reason": reason, "priority": priority}


@router.post("/{case_id}/extract-knowledge")
async def extract_knowledge(case_id: int, request: ExtractKnowledgeRequest = ExtractKnowledgeRequest()):
    """Extract a knowledge draft from a badcase."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    title = request.title or case["title"]
    if request.auto or not request.content:
        prompt = (
            "请根据以下 Badcase 信息，总结成一段可直接写入知识库的知识条目。"
            "回答应包含：问题现象、正确结论、给业主的标准话术。\n\n"
            f"标题：{case['title']}\n"
            f"描述：{case.get('description', '')}\n"
            f"证据：{case.get('evidence', '')}\n\n"
            "直接输出知识条目内容，不要添加解释。"
        )
        content, _ = await _llm_generate(prompt)
    else:
        content = request.content

    draft = db_create_knowledge_draft(
        badcase_id=case_id,
        title=title,
        content=content,
        category=request.category,
        status="draft",
    )

    # Move to fixing state if currently classified.
    if case["status"] == "classified":
        updated = db_update_badcase(case_id, status="fixing", fix_plan="extracted to knowledge draft")
        _record_action(case_id, "extract-knowledge", {"draft_id": draft["id"]}, case["status"], "fixing")
        case = updated or case

    return {"badcase": _enrich_badcase(case), "knowledge_draft": draft}


@router.post("/{case_id}/publish-draft/{draft_id}")
async def publish_knowledge_draft(case_id: int, draft_id: int):
    """Publish a knowledge draft to the official knowledge base and reindex."""
    import rag_indexer

    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="badcase not found")
    draft = db_get_knowledge_draft(draft_id)
    if not draft or draft.get("badcase_id") != case_id:
        raise HTTPException(status_code=404, detail="draft not found")

    doc = db_create_knowledge_doc(
        title=draft["title"],
        content=draft["content"],
        category=draft.get("category", "未分类"),
    )
    try:
        rag_indexer.reindex_document(doc["id"])
    except Exception as exc:
        logger.exception("reindex after publish failed")
        # Continue; the doc exists but index may be stale.

    db_update_knowledge_draft(draft_id, status="published", knowledge_doc_id=doc["id"])

    before = case["status"]
    new_status = before
    if before == "fixing":
        new_status = "verifying"
        updated = db_update_badcase(case_id, status=new_status, fix_plan="knowledge published")
        case = updated or case
    _record_action(case_id, "publish-knowledge", {"doc_id": doc["id"], "draft_id": draft_id}, before, new_status)

    return {"badcase": _enrich_badcase(case), "knowledge_doc": doc}


@router.post("/{case_id}/publish-skill-draft/{draft_id}")
async def publish_skill_prompt_draft_endpoint(
    case_id: int, draft_id: int, request: PublishSkillDraftRequest = PublishSkillDraftRequest()
):
    """Publish a skill/prompt draft to an existing skill or create a new skill."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="badcase not found")
    draft = db_get_skill_prompt_draft(draft_id)
    if not draft or draft.get("badcase_id") != case_id:
        raise HTTPException(status_code=404, detail="draft not found")

    target_skill_id = request.target_skill_id or draft.get("skill_id")
    now = datetime.now(UTC).isoformat()
    if target_skill_id:
        existing = get_skill(target_skill_id)
        # Use imported alias db_update_skill.
        db_update_skill(
            skill_id=target_skill_id,
            name=draft.get("skill_name") or (existing.get("name") if existing else f"skill-{draft['id']}"),
            description=(existing.get("description") if existing else ""),
            instructions=draft.get("prompt_content", ""),
            category=(existing.get("category") if existing else "运维"),
            enabled=True,
            trigger_condition=draft.get("trigger_keywords", ""),
        )
    else:
        new_skill = db_create_skill(
            name=draft.get("skill_name") or f"skill-{draft['id']}",
            description="",
            instructions=draft.get("prompt_content", ""),
            category="运维",
            enabled=True,
            trigger_condition=draft.get("trigger_keywords", ""),
        )
        target_skill_id = new_skill["id"]

    db_update_skill_prompt_draft(
        draft_id,
        skill_id=target_skill_id,
        status="published",
        published_at=now,
        published_by="operator",
    )

    before = case["status"]
    new_status = before
    if before == "fixing":
        new_status = "verifying"
        updated = db_update_badcase(case_id, status=new_status, fix_plan="skill/prompt published")
        case = updated or case
    _record_action(case_id, "publish-skill-prompt", {"skill_id": target_skill_id, "draft_id": draft_id}, before, new_status)

    return {"badcase": _enrich_badcase(case), "skill_id": target_skill_id}


@router.post("/{case_id}/accept-capability-gap/{draft_id}")
async def accept_capability_gap_endpoint(
    case_id: int, draft_id: int, request: AcceptGapRequest = AcceptGapRequest()
):
    """Accept a capability gap as a product backlog item; does NOT create real tools."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="badcase not found")
    draft = db_get_capability_gap_draft(draft_id)
    if not draft or draft.get("badcase_id") != case_id:
        raise HTTPException(status_code=404, detail="draft not found")

    now = datetime.now(UTC).isoformat()
    db_update_capability_gap_draft(
        draft_id,
        status="accepted",
        accepted_at=now,
        accepted_by="operator",
    )
    _record_action(
        case_id,
        "accept-capability-gap",
        {"draft_id": draft_id, "note": request.note},
        case["status"],
        case["status"],
    )
    return {"badcase": _enrich_badcase(case), "note": "能力缺口已记录为产品待办，未自动创建工具"}


def _build_price_snapshot(model_id: str) -> Optional[Dict[str, Any]]:
    price = get_enabled_price_for_model(model_id)
    if not price:
        return None
    return {
        "model_id": price.get("model_id"),
        "currency": price.get("currency"),
        "effective_date": price.get("effective_date"),
        "input_price_per_1m": price.get("input_price_per_1m"),
        "cached_input_price_per_1m": price.get("cached_input_price_per_1m"),
        "output_price_per_1m": price.get("output_price_per_1m"),
        "reasoning_price_per_1m": price.get("reasoning_price_per_1m"),
        "source_note": price.get("source_note"),
    }


def _calculate_cost(model_id: str, usage: Dict[str, Optional[int]]) -> tuple:
    snapshot = _build_price_snapshot(model_id)
    if not snapshot:
        return None, None
    input_tk = usage.get("input_tokens") or 0
    output_tk = usage.get("output_tokens") or 0
    reasoning_tk = usage.get("reasoning_tokens") or 0
    cached_tk = usage.get("cached_tokens") or 0
    cost = 0.0
    if snapshot.get("input_price_per_1m") is not None:
        cost += (input_tk - cached_tk) * (snapshot["input_price_per_1m"] / 1_000_000)
    if snapshot.get("cached_input_price_per_1m") is not None:
        cost += cached_tk * (snapshot["cached_input_price_per_1m"] / 1_000_000)
    if snapshot.get("output_price_per_1m") is not None:
        cost += output_tk * (snapshot["output_price_per_1m"] / 1_000_000)
    if snapshot.get("reasoning_price_per_1m") is not None:
        cost += reasoning_tk * (snapshot["reasoning_price_per_1m"] / 1_000_000)
    return round(cost, 8), snapshot


@router.post("/{case_id}/darwin-fix")
async def darwin_fix(case_id: int, request: DarwinFixRequest = DarwinFixRequest()):
    """Run Darwin deep analysis on a classified badcase and generate structured fix drafts."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if case["status"] != "classified":
        raise HTTPException(status_code=400, detail=f"Darwin analysis requires status=classified, got {case['status']}")

    context = case.get("context_json") or ""
    if isinstance(context, str) and context:
        try:
            context_obj = json.loads(context)
        except Exception:
            context_obj = {}
    else:
        context_obj = context or {}

    darwin = _find_darwin_skill()
    darwin_instructions = darwin.get("instructions", "") if darwin else ""
    darwin_name = darwin.get("name", "达尔文") if darwin else "达尔文"

    prompt = (
        f"你是已安装的 Darwin（达尔文）优化 Skill：{darwin_name}。\n"
        f"{darwin_instructions}\n\n"
        "请对以下 Badcase 做深度运营分析。注意：你不能自动修改代码、不能自动创建真实 MCP 工具、不能声称已完成业务操作。"
        "你只能输出分析结论和人工可审核的草稿。\n\n"
        f"标题：{case['title']}\n"
        f"分类：{case.get('category', 'other')}\n"
        f"描述：{case.get('description', '')}\n"
        f"反馈原因：{case.get('feedback_reason', '')}\n"
        f"原问题：{case.get('original_query', '')}\n"
        f"原回答：{case.get('ai_response', '')[:600]}\n"
        f"上下文：{json.dumps(context_obj, ensure_ascii=False)[:1000]}\n\n"
        "请严格输出 JSON（不要 Markdown 代码块）：\n"
        "{\n"
        '  "root_cause": "<根因分析>",\n'
        '  "suggested_actions": ["<建议动作1>", "<建议动作2>"],\n'
        '  "expected_impact": "<预期影响>",\n'
        '  "risks": "<风险说明>",\n'
        '  "drafts": [\n'
        '    {"type": "knowledge", "title": "<知识库草稿标题>", "content": "<正文>", "target_doc_title": "<目标文档名，可选>"},\n'
        '    {"type": "skill_prompt", "title": "<Skill草稿标题>", "skill_name": "<Skill名称>", "prompt_content": "<Prompt内容>", "trigger_keywords": "<触发关键词>"},\n'
        '    {"type": "capability_gap", "title": "<能力缺口标题>", "description": "<缺口描述>", "gap_type": "mcp_write|integration|data", "suggested_action": "<建议>"}\n'
        "  ]\n"
        "}\n"
    )
    if request.prompt:
        prompt = f"{request.prompt}\n\n{prompt}"

    darwin_trace_id = uuid.uuid4().hex[:16]
    model_id = "deepseek-v4-pro"
    start = time.time()
    status = "success"
    error_summary = None
    usage = {}
    try:
        analysis_text, usage = await _llm_generate(prompt, model_id=model_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        analysis_text = ""
        status = "failed"
        error_summary = str(e)[:300]
    latency_ms = int((time.time() - start) * 1000)
    total_tokens = usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    usage_source = "provider_reported" if usage.get("total_tokens") else "estimated_tokenization" if total_tokens else "unavailable"
    cost_cny, snapshot = _calculate_cost(model_id, usage)
    try:
        record_model_call(
            trace_id=darwin_trace_id,
            stage="darwin",
            model_id=model_id,
            status=status,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=usage.get("reasoning_tokens"),
            cached_tokens=usage.get("cached_tokens"),
            total_tokens=total_tokens,
            usage_source=usage_source,
            model_selection_reason="Darwin deep analysis uses Pro",
            error_summary=error_summary,
            price_snapshot=snapshot,
            estimated_cost_cny=cost_cny,
        )
    except Exception:
        pass

    analysis_obj = _extract_json(analysis_text) or {}
    if not analysis_obj and status == "success":
        analysis_obj = {
            "root_cause": "Darwin 返回无法解析",
            "suggested_actions": ["检查 Darwin 输出格式"],
            "expected_impact": "无法评估",
            "risks": "无法评估",
            "drafts": [],
        }

    # Ensure required keys exist.
    analysis_obj.setdefault("root_cause", "")
    analysis_obj.setdefault("suggested_actions", [])
    analysis_obj.setdefault("expected_impact", "")
    analysis_obj.setdefault("risks", "")
    analysis_obj.setdefault("drafts", [])

    created_drafts: List[Dict[str, Any]] = []
    for draft in analysis_obj.get("drafts", []) or []:
        draft_type = draft.get("type")
        try:
            if draft_type == "knowledge":
                created = db_create_knowledge_draft(
                    badcase_id=case_id,
                    title=draft.get("title", "未命名知识草稿"),
                    content=draft.get("content", ""),
                    category=draft.get("target_doc_title") or "未分类",
                )
                created_drafts.append({"type": "knowledge", "draft": created})
            elif draft_type == "skill_prompt":
                existing = get_skill_by_name(draft.get("skill_name", ""))
                created = db_create_skill_prompt_draft(
                    badcase_id=case_id,
                    skill_id=existing.get("id") if existing else None,
                    skill_name=draft.get("skill_name", ""),
                    title=draft.get("title", "未命名 Skill 草稿"),
                    prompt_content=draft.get("prompt_content", ""),
                    trigger_keywords=draft.get("trigger_keywords", ""),
                )
                created_drafts.append({"type": "skill_prompt", "draft": created})
            elif draft_type == "capability_gap":
                created = db_create_capability_gap_draft(
                    badcase_id=case_id,
                    title=draft.get("title", "未命名能力缺口"),
                    description=draft.get("description", ""),
                    gap_type=draft.get("gap_type", "other"),
                    suggested_action=draft.get("suggested_action", ""),
                )
                created_drafts.append({"type": "capability_gap", "draft": created})
        except Exception as exc:
            logger.exception("failed to create draft from Darwin output")
            created_drafts.append({"type": draft_type, "error": str(exc)})

    # Fallback: ensure the classified category is represented as a draft so
    # the operations loop can proceed without faking a successful model output.
    has_knowledge = any(d.get("type") == "knowledge" for d in created_drafts)
    has_capability = any(d.get("type") == "capability_gap" for d in created_drafts)
    if case.get("category") == "knowledge_gap" and not has_knowledge:
        try:
            created = db_create_knowledge_draft(
                badcase_id=case_id,
                title=f"补充：{case.get('title', '知识库缺口')[:40]}",
                content=case.get("description", ""),
                category="未分类",
            )
            created_drafts.append({"type": "knowledge", "draft": created})
        except Exception:
            logger.exception("failed to create fallback knowledge draft")
    if case.get("category") == "mcp_capability" and not has_capability:
        try:
            created = db_create_capability_gap_draft(
                badcase_id=case_id,
                title="MCP/能力缺口草稿",
                description=analysis_obj.get("root_cause") or case.get("description", ""),
                gap_type="mcp_write",
                suggested_action="待产品评估后补充对应 MCP 写操作或系统集成能力，当前不可自动完成业务操作。",
            )
            created_drafts.append({"type": "capability_gap", "draft": created})
        except Exception:
            logger.exception("failed to create fallback capability gap draft")

    before = case["status"]
    new_status = "fixing"
    updated = db_update_badcase(
        case_id,
        root_cause=analysis_obj.get("root_cause", case.get("root_cause")),
        fix_plan=json.dumps(analysis_obj.get("suggested_actions", []), ensure_ascii=False),
        darwin_analysis=json.dumps(analysis_obj, ensure_ascii=False),
        darwin_trace_id=darwin_trace_id,
        status=new_status,
    )
    _record_action(
        case_id,
        "darwin-fix",
        {
            "model_id": model_id,
            "darwin_trace_id": darwin_trace_id,
            "drafts_created": len(created_drafts),
            "analysis_keys": list(analysis_obj.keys()),
        },
        before,
        new_status,
    )
    return {
        "badcase": _enrich_badcase(updated),
        "analysis": analysis_obj,
        "drafts": created_drafts,
        "model_id": model_id,
        "darwin_skill_found": bool(darwin),
        "darwin_trace_id": darwin_trace_id,
        "usage_source": usage_source,
        "total_tokens": total_tokens,
        "estimated_cost_cny": cost_cny,
    }


@router.post("/{case_id}/retry")
async def switch_model_retry_alias(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Frontend alias for /switch-model-retry."""
    return await switch_model_retry(case_id, request)


@router.post("/{case_id}/switch-model-retry")
async def switch_model_retry(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Retry the user message with an alternative model."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    user_message = request.user_message
    if not user_message and case.get("source_message_id"):
        msg = get_chat_message(case["source_message_id"])
        if msg:
            user_message = msg.get("content", "")
    if not user_message:
        user_message = case.get("title") or ""
        if case.get("description"):
            user_message = f"{user_message}\n{case['description']}".strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message or source_message_id required")

    # Prefer explicit model_id; otherwise retry with the runtime default Flash.
    model_id = request.model_id or "deepseek-v4-flash"

    alt_model = build_model(model_id)
    prompt = (
        "你是YIAI物业物业客服助手。请专业、简洁地回答业主问题。"
        "当问题超出物业维修、收费或客服范围时，主动提出转人工。\n\n"
        f"业主问题：{user_message}"
    )
    retry_text, _ = await _llm_generate(prompt, model=alt_model)

    before = case["status"]
    new_status = "fixing" if before in ("pending", "classified") else before
    updated = db_update_badcase(
        case_id,
        status=new_status,
        fix_plan=f"model retry with {model_id}",
    )
    _record_action(case_id, "switch-model-retry", {"model_id": model_id, "response": retry_text}, before, new_status)
    return {"badcase": _enrich_badcase(updated), "model_id": model_id, "retry_response": retry_text}


@router.post("/{case_id}/verify")
async def verify_badcase(case_id: int, request: VerifyRequest = VerifyRequest()):
    """Verify the badcase fix and close or keep fixing it."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    if request.passed:
        if case["status"] != "verifying":
            raise HTTPException(status_code=400, detail="must retest before verify")
        if not case.get("retest_response"):
            raise HTTPException(status_code=400, detail="retest_response missing")
        new_status = "closed"
        updated = db_update_badcase(case_id, status=new_status, verified_by="operator")
    else:
        new_status = "fixing"
        updated = db_update_badcase(case_id, status=new_status, fix_plan=request.note or "verification failed")

    _record_action(case_id, "verify", {"passed": request.passed, "note": request.note}, case["status"], new_status)
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/transition")
async def transition_badcase(case_id: int, request: TransitionRequest = TransitionRequest()):
    """Manually transition a badcase to another valid state."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if request.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status: {request.status}")

    updated = db_update_badcase(case_id, status=request.status)
    _record_action(case_id, "transition", {"note": request.note}, case["status"], request.status, "user")
    return {"badcase": _enrich_badcase(updated)}


@router.get("/{case_id}/actions")
async def list_actions(case_id: int):
    """List lifecycle actions for a badcase."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    actions = list_badcase_actions(case_id)
    return {"actions": actions, "count": len(actions)}


@router.post("/{case_id}/darwin-optimize")
async def darwin_optimize_alias(case_id: int, request: DarwinFixRequest = DarwinFixRequest()):
    """Alias for /darwin-fix to match test-case naming."""
    return await darwin_fix(case_id, request)


@router.post("/{case_id}/darwin")
async def darwin_alias_frontend(case_id: int, request: DarwinFixRequest = DarwinFixRequest()):
    """Frontend alias for /darwin-fix."""
    return await darwin_fix(case_id, request)


@router.post("/{case_id}/close")
async def close_badcase(case_id: int, note: str = ""):
    """Close a badcase (verify passed)."""
    return await verify_badcase(case_id, VerifyRequest(passed=True, note=note))


@router.post("/{case_id}/reject")
async def reject_badcase(case_id: int, request: RejectRequest = RejectRequest()):
    """Reject a badcase with a required reason."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if not request.rejected_reason or not request.rejected_reason.strip():
        raise HTTPException(status_code=400, detail="rejected_reason required")

    new_status = "rejected"
    updated = db_update_badcase(case_id, status=new_status, rejected_reason=request.rejected_reason.strip())
    _record_action(case_id, "reject", {"reason": request.rejected_reason.strip()}, case["status"], new_status)
    return {"badcase": _enrich_badcase(updated)}


async def _consume_chat_stream(message: str, session_id: str, user_id: str = "retest") -> Dict[str, Any]:
    """Run a real chat stream and return the final answer + done context."""
    # Lazy import to avoid circular dependency between routers.
    from app.chat import _stream_agent_response

    final_answer = ""
    done_payload: Dict[str, Any] = {}
    async for chunk in _stream_agent_response(message, session_id, user_id):
        for line in chunk.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    payload = json.loads(data)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    if "content" in payload and payload.get("content"):
                        final_answer += str(payload["content"])
                    # The done event carries the full context.
                    if payload.get("status") == "complete" or "message_id" in payload:
                        done_payload = payload
    if not done_payload and final_answer:
        done_payload = {"status": "complete", "answer": final_answer}
    return {"answer": final_answer, "done": done_payload}


@router.post("/{case_id}/retest")
async def retest_badcase(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Retest the badcase user message through the real chat runtime."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if case["status"] not in ("fixing", "verifying"):
        raise HTTPException(status_code=400, detail=f"cannot retest from status {case['status']}")

    user_message = request.user_message or case.get("original_query")
    if not user_message and case.get("source_message_id"):
        msg = get_chat_message(case["source_message_id"])
        if msg:
            user_message = msg.get("content", "")
    if not user_message:
        user_message = case.get("title") or ""
        if case.get("description"):
            user_message = f"{user_message}\n{case['description']}".strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message or original_query required")

    retest_session_id = f"retest-{uuid.uuid4().hex[:12]}"
    try:
        result = await _consume_chat_stream(user_message, retest_session_id, user_id="retest")
    except Exception as e:
        logger.exception("retest chat stream failed")
        raise HTTPException(status_code=502, detail=f"retest failed: {e}")

    answer = result.get("answer", "")
    done = result.get("done", {})
    retest_context = {
        "session_id": retest_session_id,
        "route_intent": done.get("route_intent"),
        "current_agent": done.get("current_agent"),
        "activated_skills": done.get("activated_skills"),
        "citations": done.get("citations"),
        "tool_calls": done.get("tool_calls"),
        "mcp_calls": done.get("mcp_calls"),
        "model_id": done.get("model_id"),
        "token_count": done.get("token_count"),
        "trace_id": done.get("trace_id"),
        "auto_badcase_id": done.get("auto_badcase_id"),
        "usage_source": done.get("usage_source"),
    }
    retest_trace_id = done.get("trace_id")

    before = case["status"]
    new_status = "verifying"
    updated = db_update_badcase(
        case_id,
        status=new_status,
        retest_response=answer,
        retest_context_json=json.dumps(retest_context, ensure_ascii=False, default=str),
        retest_trace_id=retest_trace_id,
    )
    _record_action(
        case_id,
        "retest",
        {
            "retest_session_id": retest_session_id,
            "retest_trace_id": retest_trace_id,
            "answer_preview": answer[:200],
        },
        before,
        new_status,
    )
    return {
        "badcase": _enrich_badcase(updated),
        "retest_response": answer,
        "retest_context": retest_context,
    }


@router.post("/{case_id}/check-tools")
async def check_tools_badcase(case_id: int):
    """Analyze whether the badcase is caused by missing or misconfigured tools."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    from db.property_db import list_skills, list_mcp_servers

    enabled_skills = [s for s in list_skills() if s.get("enabled")]
    enabled_servers = [s for s in list_mcp_servers() if s.get("enabled")]
    skill_names = [s.get("name", "") for s in enabled_skills]
    tool_descriptions = []
    for server in enabled_servers:
        for tool in server.get("tools", []):
            tool_descriptions.append(f"- {server.get('name', '')}:{tool.get('name', '')} ({tool.get('description', '')})")

    prompt = (
        "你是一名 AI 工具配置审计专家。请根据以下 Badcase 信息，分析该问题是否由工具/Skill 缺失或配置错误导致。"
        "如果可能，请指出应该启用哪个 Skill 或 MCP 工具，并给出具体建议。\n\n"
        f"标题：{case['title']}\n"
        f"描述：{case.get('description', '')}\n"
        f"证据：{case.get('evidence', '')}\n\n"
        f"当前已启用 Skills：{', '.join(skill_names)}\n"
        f"当前已启用 MCP 工具：\n{chr(10).join(tool_descriptions)}\n\n"
        "请直接输出分析结论与建议，不要添加解释。"
    )
    analysis, _ = await _llm_generate(prompt)

    before = case["status"]
    new_status = "fixing" if before in ("pending", "classified") else before
    updated = db_update_badcase(
        case_id,
        status=new_status,
        fix_plan="tool configuration check: " + analysis[:200],
    )
    _record_action(case_id, "check-tools", {"analysis": analysis}, before, new_status)
    return {"badcase": _enrich_badcase(updated), "analysis": analysis}



