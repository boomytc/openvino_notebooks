# 基于Qwen2VL和OpenVINO的视觉-语言助手

Qwen2.5-VL是QwenVL系列多模态大语言模型的最新成员。

![](https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5-vl-Capybara.png)

**Qwen2.5VL的主要增强功能：**
* **视觉理解能力**：Qwen2.5-VL不仅擅长识别花、鸟、鱼、虫等常见物体，还能高效分析图像中的文本、图表、图标、图形和布局。
* **智能代理能力**：Qwen2.5-VL可直接作为视觉智能体，具备推理和动态调用工具的能力，能够操作电脑和手机。
* **长视频理解与事件捕捉**：Qwen2.5-VL可理解长达1小时以上的视频，并新增了通过定位相关视频片段来捕捉事件的能力。
* **多格式视觉定位能力**：Qwen2.5-VL可通过生成边界框或点来精确定位图像中的对象，并提供稳定JSON格式的坐标和属性输出。
* **结构化输出生成**：对于发票、表单、表格等扫描数据，Qwen2.5-VL支持其内容的结构化输出，适用于金融、商业等场景。

**模型能力：**
* **全球图像识别**：Qwen2.5-VL显著增强了通用图像识别能力，扩展了图像类别至超大规模。不仅涵盖植物、动物、名山大川地标，还包括影视IP以及各类商品。
* **精准对象定位**：Qwen2.5-VL采用边界框和点式表示进行对象定位，支持层级定位和标准化JSON输出。该增强的定位能力为视觉推理提供了基础。
* **增强的文本识别与理解**：Qwen2.5-VL的OCR识别能力提升至新水平，支持多场景、多语言、多方向的文本识别与定位。同时，信息提取能力显著增强，满足资质审核、金融业务等数字化、智能化需求。
* **强大的文档解析能力**：Qwen2.5-VL设计了独特的QwenVL HTML格式，基于HTML提取布局信息。QwenVL HTML可适用于杂志、论文、网页，甚至手机截图等多种场景的文档解析。
* **增强的视频理解能力**：Qwen2.5-VL的视频理解能力全面升级。在时间处理方面，引入了动态帧率（FPS）训练和绝对时间编码技术。因此，模型不仅支持小时级超长视频理解，还能实现秒级事件定位。可精准理解数小时长视频内容，搜索视频中的特定事件，并对不同时间段进行关键点总结，帮助用户快速高效提取视频中的关键信息。

**模型架构细节：**

与Qwen2VL相比，Qwen2.5VL架构更新如下：
* **视频理解的动态分辨率与帧率训练**
通过采用动态FPS采样，将动态分辨率扩展到时间维度，使模型能够以不同采样率理解视频。相应地，mRoPE在时间维度上更新了ID和绝对时间对齐，使模型能够学习时间序列和速度，最终获得精确定位特定时刻的能力。
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_arc.jpeg)
* **精简高效的视觉编码器**
Qwen2.5VL通过在ViT中策略性地引入窗口注意力机制，提升了训练和推理速度。ViT架构进一步优化，采用SwiGLU和RMSNorm，与Qwen2.5 LLM结构保持一致。

更多模型详情请参见[模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[博客](https://qwenlm.github.io/blog/qwen2.5-vl/)、[技术报告](https://arxiv.org/abs/2502.13923)和原始[仓库](https://github.com/QwenLM/Qwen2.5-VL)。

在本教程中，我们将介绍如何使用[Optimum Intel](https://github.com/huggingface/optimum-intel)将Qwen2.5VL模型转换和优化，以创建多模态聊天机器人。同时，我们还将演示如何使用[NNCF](https://github.com/openvinotoolkit/nncf)等技术对模型进行权重压缩等优化。

## 笔记本内容
本教程包含以下步骤：

- 安装依赖
- 转换和优化模型
- 运行OpenVINO模型推理
- 启动交互式演示

在本演示中，您将创建一个交互式聊天机器人，能够回答关于所提供图像内容的问题。

下图展示了输入提示和模型回答的示例。
![example.png](https://github.com/user-attachments/assets/7e12ac6c-12f8-43d8-9c0a-b63d6ecaf20b)

## 安装说明
这是一个自包含的示例，仅依赖于其自身代码。</br>
我们建议在虚拟环境中运行此笔记本。您只需一个Jupyter服务器即可开始。
更多详情请参见[安装指南](../../README.md)。

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen2.5-vl/README.md" />