# ADR-0011｜A 路统一人工协同合同

- 状态：Accepted
- 日期：2026-08-07
- 适用：V1.8.2 Target① A 路
- 修订：仅替代 ADR-0010 对 A 路“只表示现实安全风险”的定义；B/C 与其余边界不变

## 决策

A 路统一表示“本轮进入人工协同”，包含两个结构化子类型：

1. 业主明确要求工作人员接手：`business_intent=user_requested_handoff`，有效 Lane 必须为 `A_SAFETY_HANDOFF`，`reason_code=user_requested`，`queue=property_service`，`safety_override=false`；
2. AI 判断存在现实安全风险：有效 Lane 为 `A_SAFETY_HANDOFF`，`reason_code=safety_risk`，`queue=emergency`，`safety_override=true`。

技术枚举 `A_SAFETY_HANDOFF` 为兼容既有 Schema 暂不改名，也不迁移数据库或改写历史。产品主层统一显示“A 路：人工协同”，再按 `reason_code` 区分普通人工协同与紧急人工协同。

Router 必须根据整句语义直接把明确转人工输出为 A；不得以关键词、正则、句子白名单或测试题分支实现。为防止旧模型合同或历史状态产生 `B/C + user_requested_handoff`，运行时在 Lane SSE、AnswerContract、Evidence Ledger 和 Trace 落账之前执行结构化不变量归一。归一只消费 Router 结构化结果与已持久化 Handoff 状态，不读取或匹配用户原文，不增加 Provider 请求。

一旦有效 Lane 为 A，Agent Selector、垂直 Agent、Skill、RAG、MCP/Tool、Draft/Proposal、ActionGateway/Receipt 全部短路。普通人工协同只返回等待工作人员接手的文案；现实安全风险保留必要的紧急安全提示。Handoff 必须真实持久化，不能只改 Trace 展示。

## 原因

旧合同允许 `user_requested_handoff` 横挂在 B/C，导致 Handoff 已真实创建，但 Lane SSE、Ledger 与 Trace 仍显示 C。目标①要求“只要本轮真实进入人工协同，最终有效路径就必须是 A”，因此需要在正式 RunState 形成前收敛 Lane，而不是事后修饰页面。

## 不变约束

- B-RAG、B-Tool、B 受控写和普通 C 路不做功能改造；
- 明确转人工仍在所有普通能力与写能力之前短路；
- 业主确认受控写仍是 HITL，不等于人工接管；
- Router 每轮仍只允许一次 Provider 请求，不增加重试；
- Provider 记账、成本、时间范围、价格、数据库 Schema、RuntimeRelease 与历史 Trace 不变。

## 验收

- Fake Router 返回 `C + user_requested_handoff` 时，lane SSE、done、RunState/Ledger、Trace API 均为 A，普通队列、非安全覆盖；
- `A + safety_risk` 进入紧急队列并保留安全提示；
- 普通 C、否定转人工和咨询规则不产生 Handoff；
- A 路所有下游普通能力与写能力调用次数为零；
- 断网确定性测试 Provider 请求为零，部署后只进行一次用户授权的真实明确转人工 SSE。

## 回滚

使用普通 `git revert` 撤销本 ADR 对应实现提交，只重建 API/Web。不得 reset、force push、迁移数据库、改写历史证据或发布空 RuntimeRelease。
