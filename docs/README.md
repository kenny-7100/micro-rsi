# 文档导航

- [学习路线与验收](roadmap.md)：阶段目标、学习顺序与验收方式。
- [第一阶段实现导读](stage-one-guide.md)：从文本、张量形状到因果注意力、loss 和采样的代码阅读路径。
- [《呐喊》语料说明](corpus-nahan.md)：实际来源、清洗方式、数据统计与使用范围。
- [第一轮实验记录](stage-one-results.md)：本机训练结果、验收证据和生成局限。
- [第一阶段训练方案](stage-one-plan.md)：已确认的《呐喊》中文续写路线、起始参数、实施节点与待核实事项。
- [工程与实验约定](engineering.md)：实现方式、依赖、数据切分、检查点和验证细则。
- [经验沉淀方法](knowledge.md)：什么值得记录、放在哪里、如何维护。
- [本地环境经验](local-environment.md)：当前设备上实际遇到的问题与验证方法。

新增独立主题文档时更新此导航；稳定的全局协作规则保留在根目录 [AGENTS.md](../AGENTS.md)。

- [《彷徨》来源与成对数据](corpus-panghuang.md)：固定修订、正文提取、共享词表与限流经验。
- [第二轮对照实验](corpus-expansion-results.md)：temperature 与加入《彷徨》的同预算结果和局限。

- [百万字符语料与尺寸对照方案](wiki-readability-plan.md)：固定批量语料、清洗、隔离测试集和短续写验收。
- [百万字符语料实验结果](wiki-readability-results.md)：当前尺寸与扩大模型的实测、逐条可读性验收和局限。
- [可读性诊断与续训](training-diagnosis.md)：上下文消融、梯度和设备核验、语料缺陷及完整状态恢复。
