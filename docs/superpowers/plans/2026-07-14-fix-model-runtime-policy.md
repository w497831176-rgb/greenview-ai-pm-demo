# fix/model-runtime-policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use code-with-self-testing and nas-ssh-operations.

**Goal:** Unify runtime model policy across NAS `.env`, backend `build_model`, SQLite `model_configs`, and the frontend Models page so that owner-facing chat always uses `deepseek-v4-flash` with thinking enabled, while `deepseek-v4-pro` is reserved for A/B tests and Darwin.

**Architecture:** Move the single source of runtime truth to `DEEPSEEK_MODEL_ID` / `app.settings.MODEL` (always flash), make `build_model()` respect explicit `model_id` and default to flash, update the SQLite catalog via a startup migration, expose safe model status from `/api/models`, and simplify the frontend to read-only credential status.

**Tech Stack:** FastAPI, Agno, DeepSeek, SQLite, vanilla JS frontend, Docker Compose on Synology DSM.

---

## Task 1: Runtime Defaults

**Files:**
- Modify: `compose.yaml`
- Modify: `app/settings.py`
- Modify: `app/main.py`
- Test: `scripts/test_model_runtime.py`

- [ ] **Step 1: Change compose.yaml default**

```yaml
# before
DEEPSEEK_MODEL_ID: ${DEEPSEEK_MODEL_ID:-deepseek-v4-pro}
# after
DEEPSEEK_MODEL_ID: ${DEEPSEEK_MODEL_ID:-deepseek-v4-flash}
```

Also update the comment from "Pro 默认" to "Flash 默认运行时模型".

- [ ] **Step 2: Fix app/settings.py defaults and build_model**

Set:

```python
MODEL = "deepseek-v4-flash"
USE_THINKING = True
```

Change `build_model` so that when `model_id` is not provided it always uses `MODEL` (flash), not the SQLite default. When `model_id` is provided, use it strictly.

```python
def build_model(model_id: Optional[str] = None, **overrides) -> DeepSeek:
    resolved_id = model_id or MODEL
    # optional lookup for api_key/base_url/model_params, but never override id
    ...
    id=resolved_id,
    ...
    use_thinking=USE_THINKING,
```

- [ ] **Step 3: Ensure main.py agent defaults are flash**

Verify `_seed_agents` already sets `model_id="deepseek-v4-flash"`. No change expected unless a value still says pro.

- [ ] **Step 4: Write failing test for build_model defaults**

```python
def test_build_model_defaults():
    m = build_model()
    assert m.id == "deepseek-v4-flash"
    m2 = build_model("deepseek-v4-pro")
    assert m2.id == "deepseek-v4-pro"
    m3 = build_model("deepseek-v4-flash")
    assert m3.id == "deepseek-v4-flash"
```

- [ ] **Step 5: Run test and commit**

```bash
python -m compileall -q .
python scripts/test_model_runtime.py
```

Expected: PASS.

---

## Task 2: SQLite Model Catalog Migration

**Files:**
- Modify: `db/property_db.py`
- Test: `scripts/test_model_runtime.py`

- [ ] **Step 1: Update seed data**

In `_seed_model_configs`:

```python
(
    "deepseek-v4-flash",
    "DeepSeek V4 Flash",
    "deepseek",
    None,
    "https://api.deepseek.com",
    json.dumps({"use_thinking": True}),
    1,  # is_default
    1,
    "常规文本 Router 与垂直 Agent 主力模型",
    now, now,
),
(
    "deepseek-v4-pro",
    "DeepSeek V4 Pro",
    "deepseek",
    None,
    "https://api.deepseek.com",
    json.dumps({"use_thinking": True}),
    0,  # is_default
    1,
    "后台 A/B 与 Darwin 深度复盘模型",
    now, now,
),
```

- [ ] **Step 2: Add idempotent migration function**

Create `_migrate_model_configs(cursor)` called after `init_db` ensures the table exists. It updates existing rows to the desired state without deleting them:

```python
flash_defaults = {
    "name": "DeepSeek V4 Flash",
    "model_params": json.dumps({"use_thinking": True}),
    "is_default": 1,
    "enabled": 1,
    "description": "常规文本 Router 与垂直 Agent 主力模型",
}
pro_defaults = {
    "name": "DeepSeek V4 Pro",
    "model_params": json.dumps({"use_thinking": True}),
    "is_default": 0,
    "enabled": 1,
    "description": "后台 A/B 与 Darwin 深度复盘模型",
}
# UPDATE each row WHERE model_id matches; clear is_default on any other rows.
```

- [ ] **Step 3: Test catalog state**

```python
def test_model_catalog():
    flash = get_model_config_by_model_id("deepseek-v4-flash")
    pro = get_model_config_by_model_id("deepseek-v4-pro")
    assert flash["is_default"]
    assert not pro["is_default"]
    assert json.loads(flash["model_params"])["use_thinking"]
    assert json.loads(pro["model_params"])["use_thinking"]
```

- [ ] **Step 4: Run test and commit**

---

## Task 3: Pro-only Entry Points

**Files:**
- Modify: `app/badcases.py`
- Modify: `app/model_configs.py`
- Test: `scripts/test_model_runtime.py`

- [ ] **Step 1: Darwin uses Pro explicitly**

In `darwin_fix`:

```python
darwin_model = build_model("deepseek-v4-pro")
```

Use it for the Darwin agent/run.

- [ ] **Step 2: Switch-model retry defaults to Flash**

In `switch_model_retry_alias`, when `request.model_id` is absent, default to `"deepseek-v4-flash"` (not the old SQLite-default flipping logic). Keep the explicit override if provided.

- [ ] **Step 3: A/B test stays Flash vs Pro**

Verify `AbTestRequest` defaults to `model_a="deepseek-v4-flash"`, `model_b="deepseek-v4-pro"`. Already correct; ensure both calls pass `use_thinking=True` via `build_model`.

- [ ] **Step 4: Write tests and commit**

---

## Task 4: SSE done Event Adds Safe Model Fields

**Files:**
- Modify: `app/chat.py`
- Test: `scripts/test_model_runtime_sse.py`

- [ ] **Step 1: Capture the actual model used**

In `_stream_agent_response`, record the resolved model id and thinking flag before streaming:

```python
runtime_model_id = skill_model_id or MODEL
thinking_enabled = USE_THINKING
```

- [ ] **Step 2: Add fields to done payload**

```python
done_payload = {
    ...
    "model_id": runtime_model_id,
    "thinking_enabled": thinking_enabled,
    "model_selection_reason": "owner-facing default" if not skill_model_id else f"skill_routing:{skill_model_id}",
}
```

Do not return `reasoning_content`.

- [ ] **Step 3: Update frontend to display model info**

In `frontend/index.html`, where `event: done` is handled, store/display `model_id` and `model_selection_reason` in the process tags area.

- [ ] **Step 4: Write SSE test and commit**

```python
sse_done, _ = sse_chat(base, "装修施工允许的时间是什么？", session_id)
assert sse_done["model_id"] == "deepseek-v4-flash"
assert sse_done["thinking_enabled"] is True
```

---

## Task 5: Frontend Models Page Fix

**Files:**
- Modify: `app/models_compat.py`
- Modify: `frontend/index.html`

- [ ] **Step 1: Safe GET /api/models response**

In `models_compat.py`, transform each config to:

```python
{
    "key": cfg["model_id"],
    "model_id": cfg["model_id"],
    "name": cfg["name"],
    "is_default": cfg["is_default"],
    "thinking_enabled": json.loads(cfg.get("model_params") or "{}").get("use_thinking", False),
    "credential_status": "server_env" if os.getenv("DEEPSEEK_API_KEY") else "missing",
}
```

Never return `api_key`.

- [ ] **Step 2: Remove edit-key endpoint usage from frontend**

Delete `editModelKey` function, the `model-edit-key-btn` binding, and the button itself. Remove the API key input modal.

- [ ] **Step 3: Remove/disable set-default button**

Delete `setDefaultModel` function and the `model-set-default-btn` binding. Keep the "默认" badge for the row where `is_default` is true.

- [ ] **Step 4: Display shared credential status**

Render:

```html
<span>${m.credential_status === 'server_env' ? 'DeepSeek 服务端凭证已配置（共享）' : '服务端凭证缺失'}</span>
```

- [ ] **Step 5: Syntax check and commit**

```bash
node --check frontend/index.html  # or parse via node
```

---

## Task 6: Integration Tests

**Files:**
- Create: `scripts/test_model_runtime.py`
- Create: `scripts/test_model_runtime_sse.py`

- [ ] **Step 1: Unit-level runtime tests**

Cover `build_model`, catalog defaults, A/B endpoint contract, Darwin endpoint contract.

- [ ] **Step 2: Real SSE tests**

Three owner queries; assert `model_id=deepseek-v4-flash`, `thinking_enabled=true`.

- [ ] **Step 3: A/B test**

Call `/api/models/ab-test` and assert both `model_a.model_id` and `model_b.model_id` are present and different.

- [ ] **Step 4: Darwin test**

Create or pick a badcase, call `/api/badcases/{id}/darwin-fix`, assert response includes `model_id=deepseek-v4-pro`.

- [ ] **Step 5: Frontend model page API contract test**

Call `/api/models`; assert no `api_key` field and `credential_status` is present.

---

## Task 7: NAS Deployment

**Files:**
- Temporary: deployment script (deleted after use)

- [ ] **Step 1: Backup .env**

```bash
cp /volume3/docker/agno-demo-os/.env /volume3/docker/agno-demo-os/.env.bak.model-policy
```

- [ ] **Step 2: Update .env if needed**

If `.env` contains `DEEPSEEK_MODEL_ID=deepseek-v4-pro`, change it to `deepseek-v4-flash`. Do not print the API key.

- [ ] **Step 3: Pull main and rebuild**

```bash
docker run --rm -v /volume3/docker/agno-demo-os:/repo -w /repo alpine/git -c safe.directory=/repo pull origin main
sudo /usr/local/bin/docker compose -f /volume3/docker/agno-demo-os/compose.yaml up -d --build demo-os-api demo-os-web
sudo /usr/local/bin/docker compose -f /volume3/docker/agno-demo-os/compose.yaml restart demo-os-web
```

- [ ] **Step 4: Health checks**

```bash
curl -k https://127.0.0.1:18004/api/agents
curl -k https://127.0.0.1:18004/api/models
```

- [ ] **Step 5: Run acceptance tests on NAS**

Run `scripts/test_model_runtime.py` and `scripts/test_model_runtime_sse.py` against the NAS endpoint.

---

## Spec Coverage Check

| Requirement | Task |
|---|---|
| compose.yaml default Flash | Task 1 |
| `.env` Pro -> Flash | Task 7 |
| `app.settings.MODEL` Flash + thinking | Task 1 |
| `build_model()` respects explicit id / defaults Flash | Task 1 |
| SQLite catalog migration (flash default, pro non-default, both thinking) | Task 2 |
| Darwin uses Pro | Task 3 |
| A/B uses Flash vs Pro | Task 3 |
| SSE done adds model_id/thinking_enabled/model_selection_reason | Task 4 |
| Frontend no API key editing, no fake set-default, shared credential status | Task 5 |
| Acceptance A/B/C/D/E/F | Task 6 + 7 |
