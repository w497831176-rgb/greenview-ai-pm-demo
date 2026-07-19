# ADR-0003｜四项演示证明、显式能力绑定与优化闭环

- 状态：Accepted
- 日期：2026-07-19
- 决策范围：V1.8 产品目标、固定验收矩阵、动态扩展和 Badcase/成本闭环

## 背景

V1.8 原合同以“能力命中、动态扩展、成本治理”三项证明为主。实际面试演示还
需要把 Badcase/Evaluation 从一个失败记录入口提升为可追踪、可分析、可修复、
可复测的优化生命周期。同时，平台旧设计曾允许 RAG 不绑定 Agent；为避免多个
知识域在运行时互相干扰，当前 V1.8 已实现显式 Agent-RAG 绑定。

## 决策

1. V1.8 的完成标准升级为四项证明：
   - 能力实时命中；
   - 能力实时新增；
   - Badcase/Evaluation 能力优化闭环；
   - 能力成本治理。
2. Router Agent 是系统单例，平台只能新增垂直 Agent。Router 的职责是依据
   Published RuntimeRelease 选择垂直 Agent，不接受 Skill、MCP 或 RAG 绑定。
3. 垂直 Agent 必须在平台显式绑定 Skill、MCP 和 RAG 文档。知识文档可以独立
   创建、索引和版本化，但未进入该 Agent 发布快照绑定范围的文档不得参与检索。
4. 配置 CRUD 只产生 Draft；校验并发布 RuntimeRelease 后，仅下一新会话看到
   新能力。旧会话继续固定原 Snapshot。
5. “能力同时命中”指同一业务场景、会话和父 Trace 中，Router、垂直 Agent、
   Skill、RAG、只读 Tool、Evaluation、Trace 和 Cost 在正确阶段共同参与；
   HITL、写 Tool、人工接管和 Badcase 生成仍保留各自授权边界，可以由同一组合
   Workflow 的不同子步骤/子 Trace 验收，不得用一个无状态模型调用绕过边界。
6. 人机协同必须分别证明：
   - AI 根据风险、能力不足或用户情绪判断转人工；
   - 业主明确要求转人工。
7. Badcase/Evaluation 生命周期必须覆盖：

   ```text
   captured
   → triaged
   → root_cause_analyzed
   → solution_drafted
   → applied
   → retested
   → verified
   → closed
   ```

   并保留 duplicate、rejected、accepted_limitation 等审计分支。自动捕获与业主
   手动反馈都必须关联原消息、Trace、Evaluation 证据和后续动作记录；AI 专家
   可以提出根因与方案，但不能静默发布配置或伪造复测通过。
8. 成本优化结论必须使用同一 Evaluation 数据集比较 baseline/candidate，在质量
   门槛通过后展示模型、阶段、Provider Usage、Token、价格快照、金额、延迟和每
   成功任务成本。Provider Usage 不完整时只能证明策略与容量变化，不能宣称精确
   金额收益。

## 结果

- 固定验收矩阵以 HIT、HANDOFF、ACTION、EXT-ASR、EXT-MCP、BADCASE、COST、
  FAIL 八类用例覆盖四项目标。
- 平台中“新增 Agent”只出现垂直 Agent；Skill、MCP、RAG 均由垂直 Agent 手动
  绑定并通过发布快照生效。
- Badcase 页面展示的不再只是问题列表，而是从捕获、归因、方案、应用、复测到
  关闭的证据链。
- 任何“已完成”声明必须分别说明：契约测试、真实后端运行、人工 UI 验收和部署
  状态，不能互相替代。

