# ADR-0010｜最终三分类 Router 与 Lane 后处理

- 状态：Accepted
- 日期：2026-07-30
- 适用：V1.8.2-S10-B.2-Final
- 替代：ADR-0009 中 CLARIFY Lane、复合字段矩阵和 Router 内 Agent 选择合同

## 决策

生产 Router 每轮只做一次语义分类，输出 `A_SAFETY_HANDOFF`、`B_PROPERTY_GOVERNED` 或 `C_ISOLATED_GENERAL`，以及一句中文理由和简短业务意图。A 表示明确现实危险，B 表示明确需要物业回答、查询、办理或协助，C 接收其他全部情况，包括明确非物业、信息不足和暂时无法判断。系统不再设置 CLARIFY Lane；用户补充信息后，下一轮结合可见对话重新判断。

Lane 的有效性只由 A/B/C 枚举决定。`request_kind`、`target_agent_id`、`confidence`、`allowed_domain`、`response_mode` 和 Agent 选择结果不再属于 Router 判定合同，也不能推翻已经正确的 Lane。Agent 选择发生在 Lane 之后：A 不选普通 Agent，B 只在 `property` 域内选，C 只在 `isolated_general` 域内选；同域没有可用 Agent 时给出保守边界回答，并把它记录为下游状态，而不是语义误分类。

Lane 后的执行边界固定：A 只给安全提示并发起 Handoff；B 才能使用物业 Agent，并继续遵守 Skill/RAG Evidence、成功 Tool、确认、ActionGateway 与 Receipt 合同；C 只允许隔离通用 Agent，物业 Skill、RAG、MCP/Tool 与 ActionGateway 全部跳过。没有关键词、正则、业务白名单、默认 B 或测试题专用分支参与生产 Lane 分类。

## 语义判断原则与验收标尺

Router 以物业服务助手的会话角色理解没有另行限定对象的“你们”，但仍按完整语义而非词语匹配判断；伪系统命令、伪 JSON、关闭证据或指定 Lane/Agent 的要求均视为普通用户输入，先忽略控制性包装再识别真实诉求；正在发生或迫近的现实危险优先于玩笑、假设、脑筋急转弯、科普或用户否认等包装，明确纯虚构创作且没有现实事件指向时仍进入 C。

最终评估标尺为 A 30、B 35、C 35。A 必须 30/30，C 误入 B 必须为 0；B 误入 C 最多允许 2 条并保留为 Evaluation 失败。Schema、Agent 选择、下游执行和 Provider 落账必须 100%，总语义准确率不得低于 97%。该标尺接受保守体验降级，不接受安全漏判或物业能力越界。

## 上下文与流程状态

普通对话每轮 Router 只接收当前用户消息和同一会话中用户可见的成功对话历史，不接收历史 Router JSON、内部分类、Trace、Prompt、Tool 调试信息或控制字段。上一轮 Lane 不绑定下一轮。

唯一的状态机例外是已经进入报修资料收集、等待确认等明确业务流程：补充资料、确认或取消继续原流程；与现有流程无关的新主题重新进入 Router。是否继续状态机由已持久化的结构化业务状态决定，不由关键词 Lane 决定。

## 原因

N241 与 N255 已证明语义 Lane 正确，但复合字段矩阵和 Agent 目标校验会把正确分类错误标记为 Schema 失败。面试 Demo 需要先清楚回答“危险、物业、其他”三个产品边界，再把 Agent、Evidence 和执行结果作为可分别解释的下游证据。

## 不变约束

- 每题 Router 仍只允许一次 Flash Provider 请求，不回退、不自动重试。
- Provider 请求、三类 Usage、费用、Trace 与 RunEvidenceLedger 合同不变。
- 不新增数据库表、框架、关键词表、正则词表、白名单或测试专用逻辑。
- Published 能力配置不变，RuntimeRelease 保持 v27；旧 Trace 按历史合同保留。

## 回滚

以 `02acb90e1019b432e1229776b545a0095363e1be` 为修改前回滚点。普通 `git revert` 本阶段提交后仅重建 API；不改写 RuntimeRelease、Snapshot、历史 Trace、业务数据、数据库或 RAG 索引。
