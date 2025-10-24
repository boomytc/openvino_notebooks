# 使用Qwen3和OpenVINO进行文本嵌入和重排序

Qwen3 Embedding模型系列是通义千问家族最新的专有模型，专门设计用于文本嵌入和排序任务。该系列基于Qwen3系列的密集基础模型，提供多种规模（0.6B、4B和8B）的文本嵌入和重排序模型。该系列继承了其基础模型卓越的多语言能力、长文本理解能力和推理能力。Qwen3 Embedding系列在多个文本嵌入和排序任务中取得了显著进展，包括文本检索、代码检索、文本分类、文本聚类和双语文本挖掘。

<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen_embedding.png" width="400"/>

在本教程中，我们将探讨如何使用OpenVINO转换和优化Qwen3 Embedding和重排序模型。

### 笔记本内容

本教程包含以下步骤：

- 前置条件
- 选择模型
- 使用Optimum Intel转换模型
- 使用Optimum-intel运行OpenVINO模型推理

## 安装说明

这是一个自包含的示例，仅依赖于其自身的代码。</br>
我们建议在虚拟环境中运行此笔记本。您只需要一个Jupyter服务器即可开始。
更多详细信息，请参阅[安装指南](../../README.md)。
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen3-embedding/README.md" />