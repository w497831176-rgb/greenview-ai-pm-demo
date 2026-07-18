# YIAI物业｜V1.7.1 运行时收敛交接包

## 用途

本资料包服务于一次**新的 AI Coding 会话**。它不是产品宣传稿，也不是重新罗列技术名词；它的目标是让接手者先理解当前系统为什么会在多技术栈联动时失稳，再提出收敛设计。

项目是求职 AI 产品经理的个人演示项目：需要真实、可解释、可验证，但不需要为生产级并发、SLA、完整安全合规或运维仪式牺牲演示价值。

## 冻结基线

- 仓库：`https://github.com/w497831176-rgb/greenview-ai-pm-demo`
- 基线：`main` / annotated tag `v1.7.1`
- 基线提交：`c531d97e582fe9d7feb567701d3b9c391bb5e17b`
- 本交接包所在分支：`docs/v1.7.1-runtime-convergence-handover`（故默认 `main` clone 不会自动包含这些文档）。
- 本次交接时的代码状态：V1.7.1 已修复“完整报修字段未进入工作流”“确认提交未真建单”“动态儿童教育 Skill 未命中”三项运行时契约。
- 不应把任何旧版本（V1.2～V1.7.0）的展示描述当成当前事实；先以此 tag、代码和实际数据库配置为准。

## 阅读顺序

1. [01-product-objective-and-demo-contract.md](01-product-objective-and-demo-contract.md)：项目究竟要证明什么。
2. [02-current-runtime-map.md](02-current-runtime-map.md)：当前请求如何穿过 Agent、Skill、RAG、MCP、工作流、Trace。
3. [03-known-failure-modes-and-root-causes.md](03-known-failure-modes-and-root-causes.md)：为什么已经修了很多轮仍会回归。
4. [04-target-runtime-contract-and-golden-flows.md](04-target-runtime-contract-and-golden-flows.md)：下一版应该收敛成哪三条演示路径。
5. [05-deployment-data-and-safety-boundaries.md](05-deployment-data-and-safety-boundaries.md)：NAS、数据卷、Git 与部署边界。
6. [06-manual-demo-and-acceptance-matrix.md](06-manual-demo-and-acceptance-matrix.md)：用户如何手点、接手者如何低成本验收。
7. [07-new-coding-session-bootstrap.md](07-new-coding-session-bootstrap.md)：可直接复制给新会话的完整提示词。
8. [08-v1.8-enterprise-runtime-architecture-contract.md](08-v1.8-enterprise-runtime-architecture-contract.md)：V1.8 当前批准的 Living Architecture；它允许通过 ADR 和版本治理持续修订，但实现不得无记录地偏离。

V1.8 实施接续时还必须读取：

- [ADR-0002｜显式发布、会话快照与组合验收](../../adr/0002-runtime-release-session-snapshot-and-composite-workflow.md)；
- [V1.8.0 发布记录](../../releases/v1.8.0.md)，其中严格区分工作区实现、契约验证、真实模型验证、人工 UI 验收和 NAS 部署。

## 绝对优先级

系统只需要稳定地证明三件事：

1. **常见 AI 技术栈真实命中且可解释**：Agent 路由、Skill、RAG、MCP/Tool、人机协同、Badcase/Evaluation、Trace、成本。
2. **能力能动态扩展**：新增 Agent、Skill、MCP 后，它们不是后台摆设，而是能实际参与下一次运行时链路。
3. **成本治理能解释且能落地**：模型选择、Token 用量、单价快照、不可得数据、优化措施和预算策略说清楚。

不是优先级的内容：生产 SLA、海量压测、复杂 CI、浏览器自动化、为演示无关的依赖升级、未经用户确认的清理/重置。
