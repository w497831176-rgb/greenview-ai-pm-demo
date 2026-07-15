"""
Badcase Closed-Loop API
=======================

Implements the full badcase lifecycle:
    pending -> classified -> fixing -> verifying -> closed/rejected

Supports automatic classification, knowledge extraction, Darwin skill
optimization, model switch retry, and verification.
"""

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.settings import MODEL, build_model
from db.property_db import (
    add_badcase_action,
    create_badcase as db_create_badcase,
    get_enabled_price_for_model,
    record_model_call,
    create_knowledge_doc as db_create_knowledge_doc,
    create_knowledge_draft as db_create_knowledge_draft,
    delete_badcase as db_delete_badcase,
    get_badcase as db_get_badcase,
    get_chat_message,
    get_knowledge_draft as db_get_knowledge_draft,
    get_skill_by_name,
    list_badcase_actions,
    list_badcases as db_list_badcases,
    list_knowledge_drafts as db_list_knowledge_drafts,
    list_skills,
    update_badcase as db_update_badcase,
    update_knowledge_draft as db_update_knowledge_draft,
)

router = APIRouter(tags=["badcases"])

VALID_CATEGORIES = {"knowledge", "skill", "model", "tool", "other"}
VALID_STATUSES = {"pending", "classified", "fixing", "verifying", "closed", "rejected"}


def _enrich_badcase(badcase: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add frontend-compatible aliases to a badcase record."""
    if not badcase:
        return None
    enriched = dict(badcase)
    enriched["query"] = enriched.get("title") or "-"
    enriched["ai_response"] = enriched.get("description") or "-"
    enriched["feedback_reason"] = enriched.get("rejected_reason") or "-"
    enriched["analysis_category"] = enriched.get("category") or "-"
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


class BadcaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None
    root_cause: Optional[str] = None
    fix_plan: Optional[str] = None
    rejected_reason: Optional[str] = None


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


class TransitionRequest(BaseModel):
    status: str = "verifying"
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


async def _llm_generate(prompt: str, model: Optional[Any] = None) -> Tuple[str, Dict[str, Optional[int]]]:
    """Generate text using the default or a provided model."""
    from agno.agent import Agent

    agent = Agent(model=model or MODEL, markdown=False)
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
    """Auto-classify a badcase into knowledge/skill/model/tool/other."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    if request.auto:
        prompt = (
            "你是一名 AI 问题分类专家。请根据下面的 Badcase 信息，从以下类别中选择一个最贴切的："
            "knowledge（知识缺失/错误）、skill（Skill 指令缺陷）、model（模型回复质量差）、"
            "tool（工具调用失败/错误）、other（其他）。\n\n"
            f"标题：{case['title']}\n"
            f"描述：{case.get('description', '')}\n"
            f"证据：{case.get('evidence', '')}\n\n"
            "请严格输出 JSON：{\"category\": \"<类别>\", \"reason\": \"<一句话理由>\"}"
        )
        raw = await _llm_generate(prompt)
        parsed = _extract_json(raw) or {}
        category = parsed.get("category", "other")
        reason = parsed.get("reason", "自动分类失败，归入 other")
        if category not in VALID_CATEGORIES:
            category = "other"
    else:
        category = request.category or "other"
        reason = request.reason
        if category not in VALID_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"invalid category: {category}")

    new_status = "classified"
    updated = db_update_badcase(case_id, category=category, status=new_status)
    _record_action(case_id, "classify", {"category": category, "reason": reason, "raw": raw if request.auto else None}, case["status"], new_status)
    return {"badcase": _enrich_badcase(updated), "reason": reason}


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
        content = await _llm_generate(prompt)
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
    """Publish a knowledge draft to the official knowledge base."""
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
    db_update_knowledge_draft(draft_id, status="published")

    # If fixing, move to verifying after publishing knowledge.
    if case["status"] == "fixing":
        updated = db_update_badcase(case_id, status="verifying", fix_plan="knowledge published")
        _record_action(case_id, "publish-knowledge", {"doc_id": doc["id"]}, case["status"], "verifying")
        case = updated or case

    return {"badcase": _enrich_badcase(case), "knowledge_doc": doc}


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
    """Call the Darwin skill to optimize the badcase fix plan."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    darwin = _find_darwin_skill()
    darwin_instructions = darwin.get("instructions", "") if darwin else ""
    darwin_name = darwin.get("name", "达尔文") if darwin else "达尔文"

    prompt = (
        f"你是已安装的 Darwin（达尔文）优化 Skill：{darwin_name}。\n"
        f"{darwin_instructions}\n\n"
        "请根据以下 Badcase 生成修复方案，包括：根因分析、修复动作、验证方法。\n\n"
        f"标题：{case['title']}\n"
        f"描述：{case.get('description', '')}\n"
        f"证据：{case.get('evidence', '')}\n"
        f"当前分类：{case.get('category', 'other')}\n\n"
        "直接输出修复方案，不要添加解释。"
    )
    if request.prompt:
        prompt = f"{request.prompt}\n\n{prompt}"

    trace_id = uuid.uuid4().hex[:16]
    model_id = "deepseek-v4-pro"
    start = time.time()
    status = "success"
    error_summary = None
    usage = {}
    try:
        # Darwin deep-fix always runs on Pro; normal classify/retest stay on Flash.
        darwin_model = build_model(model_id)
        fix_plan, usage = await _llm_generate(prompt, model=darwin_model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        fix_plan = ""
        status = "failed"
        error_summary = str(e)[:300]
    latency_ms = int((time.time() - start) * 1000)
    total_tokens = usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    usage_source = "provider_reported" if usage.get("total_tokens") else "estimated_tokenization" if total_tokens else "unavailable"
    cost_cny, snapshot = _calculate_cost(model_id, usage)
    try:
        record_model_call(
            trace_id=trace_id,
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
            model_selection_reason="Darwin deep-fix uses Pro",
            error_summary=error_summary,
            price_snapshot=snapshot,
            estimated_cost_cny=cost_cny,
        )
    except Exception:
        pass

    # Move from classified -> fixing.
    before = case["status"]
    new_status = "fixing" if before in ("pending", "classified") else before
    updated = db_update_badcase(
        case_id,
        root_cause="待达尔文优化后补充" if not case.get("root_cause") else case["root_cause"],
        fix_plan=fix_plan,
        status=new_status,
    )
    _record_action(
        case_id,
        "darwin-fix",
        {"fix_plan": fix_plan, "skill_used": bool(darwin), "model_id": model_id, "trace_id": trace_id},
        before,
        new_status,
    )
    return {
        "badcase": _enrich_badcase(updated),
        "fix_plan": fix_plan,
        "model_id": model_id,
        "darwin_skill_found": bool(darwin),
        "trace_id": trace_id,
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
    retry_response = await _llm_generate(prompt, model=alt_model)

    before = case["status"]
    new_status = "fixing" if before in ("pending", "classified") else before
    updated = db_update_badcase(
        case_id,
        status=new_status,
        fix_plan=f"model retry with {model_id}",
    )
    _record_action(case_id, "switch-model-retry", {"model_id": model_id, "response": retry_response}, before, new_status)
    return {"badcase": _enrich_badcase(updated), "model_id": model_id, "retry_response": retry_response}


@router.post("/{case_id}/verify")
async def verify_badcase(case_id: int, request: VerifyRequest = VerifyRequest()):
    """Verify the badcase fix and close or reject it."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    if request.passed:
        new_status = "closed"
        updated = db_update_badcase(case_id, status=new_status, verified_by="system")
    else:
        new_status = "rejected"
        updated = db_update_badcase(case_id, status=new_status, rejected_reason=request.note or "verification failed")

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
async def reject_badcase(case_id: int, note: str = ""):
    """Reject a badcase (verify failed)."""
    return await verify_badcase(case_id, VerifyRequest(passed=False, note=note or "rejected by user"))


@router.post("/{case_id}/retest")
async def retest_badcase(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Retest the badcase user message with the current default model."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    user_message = request.user_message
    if not user_message and case.get("source_message_id"):
        msg = get_chat_message(case["source_message_id"])
        if msg:
            user_message = msg.get("content", "")
    if not user_message:
        # Fallback to reconstructing the user message from the badcase title/description.
        user_message = case.get("title") or ""
        if case.get("description"):
            user_message = f"{user_message}\n{case['description']}".strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message or source_message_id required")

    prompt = (
        "你是YIAI物业物业客服助手。请专业、简洁地回答业主问题。"
        "当问题超出物业维修、收费或客服范围时，主动提出转人工。\n\n"
        f"业主问题：{user_message}"
    )
    retest_response = await _llm_generate(prompt, model=MODEL)

    before = case["status"]
    new_status = "fixing" if before in ("pending", "classified") else before
    updated = db_update_badcase(
        case_id,
        status=new_status,
        fix_plan="retest with current default model",
    )
    _record_action(case_id, "retest", {"response": retest_response}, before, new_status)
    return {"badcase": _enrich_badcase(updated), "retest_response": retest_response}


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
    analysis = await _llm_generate(prompt)

    before = case["status"]
    new_status = "fixing" if before in ("pending", "classified") else before
    updated = db_update_badcase(
        case_id,
        status=new_status,
        fix_plan="tool configuration check: " + analysis[:200],
    )
    _record_action(case_id, "check-tools", {"analysis": analysis}, before, new_status)
    return {"badcase": _enrich_badcase(updated), "analysis": analysis}



