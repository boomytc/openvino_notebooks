# 基于 Qwen2.5-Omni 和 OpenVINO 的全模态助手

Qwen2.5-Omni 是一个端到端的多模态模型，能够感知多种模态，包括文本、图像、音频和视频，同时以流式方式生成文本和自然语音响应。

![Image](https://github.com/user-attachments/assets/600798db-c80e-4d1d-ab00-fa945cdcd583)

有关模型的更多详情，请参见 [模型卡](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) 和原始 [仓库](https://github.com/QwenLM/Qwen2.5-Omni)。

在本教程中，我们将介绍如何转换和优化 Qwen2.5-Omni 模型，以创建全模态聊天机器人。此外，我们还将演示如何在大语言模型（LLM）部分应用有状态转换，以及使用 [NNCF](https://github.com/openvinotoolkit/nncf) 进行模型优化技术，如权重压缩。

## 笔记本内容
本教程包含以下步骤：

- 安装依赖项
- 下载 PyTorch 模型
- 将模型转换为 OpenVINO 中间表示（IR）
- 压缩语言模型权重
- 运行 OpenVINO 模型推理
- 启动交互式演示

在本演示中，您将创建一个交互式聊天机器人，它可以回答关于所提供图像内容的问题。下图展示了模型的工作结果。
![Image](https://github.com/user-attachments/assets/83e1e0f7-1a12-426b-b3f8-794662812cd4)

## 安装说明
这是一个自包含的示例，仅依赖于其自身的代码。</br>
我们建议在虚拟环境中运行此笔记本。您只需一个 Jupyter 服务器即可开始。
更多详情，请参见 [安装指南](../../README.md)。

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen2.5-omni-chatbot/README.md" />