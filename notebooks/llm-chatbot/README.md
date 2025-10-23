# 使用OpenVINO创建LLM驱动的聊天机器人

在人工智能（AI）快速发展的世界中，聊天机器人已成为企业增强客户互动和优化运营的强大工具。  
大型语言模型（LLMs）是能够理解和生成人类语言的人工智能系统。它们利用深度学习算法和海量数据来学习语言的细微差别，并生成连贯且相关的响应。  
虽然一个基本的基于意图的聊天机器人可以回答订单管理、常见问题和政策问题等简单的一次性查询，但LLM聊天机器人可以处理更复杂、多轮次的问题。LLM使聊天机器人能够通过上下文记忆以类似人类的方式进行对话式支持。借助语言模型的能力，聊天机器人正变得越来越智能，能够以极高的准确性理解和回应人类语言。

之前，我们已经讨论了如何使用OpenVINO构建指令遵循管道，请参阅[LLM问答示例](../llm-question-answering)以获取参考。  
在本教程中，我们将探讨如何利用OpenVINO的强大功能来运行大型语言模型，用于聊天和文档问答。我们将使用来自[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)库的预训练模型。

本笔记本系列包括两种创建推理管道的演示：

* [使用Optimum Intel的LLM聊天机器人](./llm-chatbot.ipynb)。使用[Hugging Face Optimum Intel库](https://huggingface.co/docs/optimum/intel/inference)创建管道，以OpenVINO Runtime运行推理。

* [使用Generate API的LLM聊天机器人](./llm-chatbot-generate-api.ipynb)。使用[OpenVINO Generate API](https://github.com/openvinotoolkit/openvino.genai)生成指令遵循推理管道。

本教程支持多种模型，您可以从提供的选项中选择一个，以比较开源LLM解决方案的质量。  
>**注意**：某些模型的转换可能需要用户进行额外操作，并且至少需要64GB RAM进行转换。

可用选项包括：

* **tiny-llama-1b-chat** - 该模型是在[TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)基础上微调的聊天模型。TinyLlama项目旨在使用与Llama 2相同的架构和分词器，在3万亿个标记上预训练一个1.1B的Llama模型。这意味着TinyLlama可以轻松集成到许多基于Llama的开源项目中。此外，TinyLlama仅包含1.1B个参数，非常紧凑。这种紧凑性使其能够满足对计算和内存占用有严格要求的多种应用。更多模型详情请参见[model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)  
* **mini-cpm-2b-dpo** - MiniCPM是由ModelBest Inc.和TsinghuaNLP开发的端到端LLM，不包括嵌入层仅含2.4B个参数。经过直接偏好优化（DPO）微调后，MiniCPM在许多流行的7B、13B和70B模型中表现更优。更多详情请参见[model_card](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16)。  
* **llama-3.2-1B-instruct** - 来自LLama3.2系列的1B参数指令调优多语言模型。Llama 3.2指令调优纯文本模型针对多语言对话使用场景进行了优化，包括代理检索和摘要任务。它们在常见的行业基准测试中表现优于许多现有的开源和闭源聊天模型。更多详情请参见[model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **llama-3.2-3B-instruct** - 来自LLama3.2系列的3B参数指令调优多语言模型。Llama 3.2指令调优纯文本模型针对多语言对话使用场景进行了优化，包括代理检索和摘要任务。它们在常见的行业基准测试中表现优于许多现有的开源和闭源聊天模型。更多详情请参见[model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **gemma-2b-it** - Gemma是Google推出的一系列轻量级、最先进的开源模型，基于创建Gemini模型的相同研究和技术。它们是文本到文本、仅解码器的大型语言模型，提供英文版本，具有开放权重、预训练变体和指令调优变体。Gemma模型适用于多种文本生成任务，包括问答、摘要和推理。此模型是2B参数模型的指令调优版本。更多模型详情请参见[model card](https://huggingface.co/google/gemma-2b-it)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/google/gemma-2b-it)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **gemma-2-2b-it** - Gemma2是Google推出的Gemma系列第二代轻量级、最先进的开源模型，基于创建Gemini模型的相同研究和技术。它们是文本到文本、仅解码器的大型语言模型，提供英文版本，具有开放权重、预训练变体和指令调优变体。Gemma模型适用于多种文本生成任务，包括问答、摘要和推理。此模型是2B参数模型的指令调优版本。更多模型详情请参见[model card](https://huggingface.co/google/gemma-2-2b-it)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/google/gemma-2-2b-it)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **red-pajama-3b-chat** - 基于GPT-NEOX架构的2.8B参数预训练语言模型。由Together Computer和开源AI社区的领导者开发。该模型在OASST1和Dolly2数据集上进行了微调，以增强聊天能力。更多模型详情请参见[HuggingFace模型卡](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)。  
* **phi3-mini-instruct** - Phi-3-Mini是一个3.8B参数的轻量级、最先进的开源模型，使用包含合成数据和经过筛选的公开网站数据的Phi-3数据集进行训练，重点在于高质量和密集推理属性。更多模型详情请参见[model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)、[Microsoft博客](https://aka.ms/phi3blog-april)和[技术报告](https://aka.ms/phi3-tech-report)。  
* **phi-3.5-mini-instruct** - Phi-3.5-mini是一个轻量级、最先进的开源模型，基于Phi-3所使用的数据集——合成数据和经过筛选的公开网站数据，重点在于极高质量、密集推理数据。该模型属于Phi-3模型系列，支持128K的上下文长度。该模型经过严格的增强过程，结合了监督微调、近端策略优化和直接偏好优化，以确保精确的指令遵循和强大的安全措施。更多模型详情请参见[model card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)、[Microsoft博客](https://aka.ms/phi3.5-techblog)和[技术报告](https://arxiv.org/abs/2404.14219)。  
* **phi-4-mini-instruct** - Phi-4-mini是一个轻量级、最先进的开源模型，基于合成数据集和经过筛选的公共领域网站数据的混合，重点在于高质量、密集推理数据。更多模型详情请参见[model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)。  
* **phi-4** - Phi-4是一个14B参数的模型，基于合成数据集、经过筛选的公共领域网站数据、获取的学术书籍和问答数据集的混合。该方法的目标是确保小型能力模型在高质量和高级推理数据上进行训练。Phi-4经过严格的增强和对齐过程，结合了监督微调和直接偏好优化，以确保精确的指令遵循和强大的安全措施。更多模型详情请参见[model_card](https://huggingface.co/microsoft/phi-4)、[技术报告](https://arxiv.org/pdf/2412.08905)和[Microsoft博客](https://techcommunity.microsoft.com/blog/aiplatformblog/introducing-phi-4-microsoft%E2%80%99s-newest-small-language-model-specializing-in-comple/4357090)。  
* **phi-4-mini-reasoning** - Phi-4-mini-reasoning是一个轻量级开源模型，基于合成数据，重点在于高质量、密集推理数据，并进一步微调以增强更高级的数学推理能力。更多模型详情请参见[model card](https://huggingface.co/microsoft/Phi-4-mini-reasoning)。  
* **phi-4-reasoning** - Phi-4-reasoning是一个最先进的开源权重推理模型，基于Phi-4，使用链式思维轨迹数据集进行监督微调，并使用强化学习进行微调。监督微调数据集包括合成提示和来自公共领域网站的高质量筛选数据，重点在于数学、科学和编码技能，以及安全和负责任AI的对齐数据。该方法的目标是确保小型能力模型在高质量和高级推理数据上进行训练。更多模型详情请参见[model card](https://huggingface.co/microsoft/Phi-4-reasoning)。  
* **gemma-7b-it** - Gemma是Google推出的一系列轻量级、最先进的开源模型，基于创建Gemini模型的相同研究和技术。它们是文本到文本、仅解码器的大型语言模型，提供英文版本，具有开放权重、预训练变体和指令调优变体。Gemma模型适用于多种文本生成任务，包括问答、摘要和推理。此模型是7B参数模型的指令调优版本。更多模型详情请参见[model card](https://huggingface.co/google/gemma-7b-it)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/google/gemma-7b-it)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **gemma-2-9b-it** - Gemma2是Google推出的Gemma系列第二代轻量级、最先进的开源模型，基于创建Gemini模型的相同研究和技术。它们是文本到文本、仅解码器的大型语言模型，提供英文版本，具有开放权重、预训练变体和指令调优变体。Gemma模型适用于多种文本生成任务，包括问答、摘要和推理。此模型是9B参数模型的指令调优版本。更多模型详情请参见[model card](https://huggingface.co/google/gemma-2-9b-it)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/google/gemma-2-9b-it)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **llama-2-7b-chat** - LLama 2是Meta开发的LLama模型的第二代。Llama 2是一系列预训练和微调的生成文本模型，参数规模从70亿到700亿不等。llama-2-7b-chat是LLama 2的70亿参数版本，针对对话使用场景进行了微调和优化。更多模型详情请参见[paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)、[repository](https://github.com/facebookresearch/llama)和[HuggingFace模型卡](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **llama-3-8b-instruct** - Llama 3是一个自回归语言模型，使用优化的Transformer架构。调优版本使用监督微调（SFT）和人类反馈强化学习（RLHF）来对齐人类对帮助性和安全性的偏好。Llama 3指令调优模型针对对话使用场景进行了优化，在常见的行业基准测试中表现优于许多现有的开源聊天模型。更多模型详情请参见[Meta博客](https://ai.meta.com/blog/meta-llama-3/)、[模型网站](https://llama.meta.com/llama3)和[model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **llama-3.1-8b-instruct** - Llama 3.1指令调优纯文本模型（8B、70B、405B）针对多语言对话使用场景进行了优化，在常见的行业基准测试中表现优于许多现有的开源和闭源聊天模型。更多模型详情请参见[Meta博客](https://ai.meta.com/blog/meta-llama-3-1/)、[模型网站](https://llama.meta.com)和[model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **qwen2.5-0.5b-instruct/qwen2.5-1.5b-instruct/qwen2.5-3b-instruct/qwen2.5-7b-instruct/qwen2.5-14b-instruct** - Qwen2.5是Qwen大语言模型的最新系列。与Qwen2相比，Qwen2.5系列在编码、数学和通用知识技能方面有显著提升。此外，它还支持长上下文和多种语言，包括中文、英文、法语、西班牙语、葡萄牙语、德语、意大利语、俄语、日语、韩语、越南语、泰语、阿拉伯语等。  
更多详情请参见[model_card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[博客](https://qwenlm.github.io/blog/qwen2.5/)、[GitHub](https://github.com/QwenLM/Qwen2.5)和[文档](https://qwen.readthedocs.io/en/latest/)。  
* **qwen-7b-chat** - Qwen-7B是阿里云提出的大型语言模型系列Qwen（简称通义千问）的7B参数版本。Qwen-7B是一个基于Transformer的大型语言模型，使用大量数据（包括网络文本、书籍、代码等）进行预训练。更多关于Qwen的详情请参见[GitHub](https://github.com/QwenLM/Qwen)代码库。  
* **mpt-7b-chat** - MPT-7B是MosaicPretrainedTransformer（MPT）模型系列的一部分，使用经过修改的Transformer架构，以优化高效训练和推理。这些架构修改包括性能优化的层实现，以及通过用Attention with Linear Biases ([ALiBi](https://arxiv.org/abs/2108.12409)) 替换位置嵌入来消除上下文长度限制。由于这些修改，MPT模型可以以高吞吐量效率和稳定收敛进行训练。MPT-7B-chat是一个用于对话生成的聊天机器人式模型。它通过在ShareGPT-Vicuna、[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)、[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)、[HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)和Evol-Instruct数据集上微调MPT-7B构建而成。更多模型详情请参见[博客](https://www.mosaicml.com/blog/mpt-7b)、[代码库](https://github.com/mosaicml/llm-foundry/)和[HuggingFace模型卡](https://huggingface.co/mosaicml/mpt-7b-chat)。  
* **chatglm3-6b** - ChatGLM3-6B是ChatGLM系列的最新开源模型。在保留前两代的许多优秀特性（如流畅对话和低部署门槛）的同时，ChatGLM3-6B采用了更多样化的训练数据集、更充分的训练步骤和更合理的训练策略。ChatGLM3-6B采用新设计的[Prompt格式](https://github.com/THUDM/ChatGLM3/blob/main/PROMPT_en.md)，除了常规的多轮对话外。您可以在[model card](https://huggingface.co/THUDM/chatglm3-6b)中找到更多模型详情。  
* **mistral-7b** - Mistral-7B-v0.1大型语言模型（LLM）是一个具有70亿参数的预训练生成文本模型。您可以在[model card](https://huggingface.co/mistralai/Mistral-7B-v0.1)、[paper](https://arxiv.org/abs/2310.06825)和[发布博客](https://mistral.ai/news/announcing-mistral-7b/)中找到更多模型详情。  
* **mistral-7B-Instruct-v0.3** - Mistral-7B-Instruct-v0.3是当前最先进的大型语言模型（LLM），适用于各种语言理解和生成任务，它是Mistral-7B-v0.3的指令微调版本。您可以在[model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)中找到更多模型详情。  
>**注意**：运行模型演示，您需要接受许可协议。  
>您必须是🤗 Hugging Face Hub的注册用户。请访问[HuggingFace模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，仔细阅读使用条款并点击接受按钮。您需要在以下代码中使用访问令牌。有关访问令牌的更多信息，请参见[文档的此部分](https://huggingface.co/docs/hub/security-tokens)。  
* **zephyr-7b-beta** - Zephyr是一系列被训练为有用助手的语言模型。Zephyr-7B-beta是该系列的第二个模型，是[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)的微调版本，使用[直接偏好优化（DPO）](https://arxiv.org/abs/2305.18290)在公开可用和合成数据集的混合上进行训练。您可以在[技术报告](https://arxiv.org/abs/2310.16944)和[HuggingFace模型卡](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)中找到更多模型详情。  
* **neural-chat-7b-v3-1** - 使用Intel Gaudi对Mistral-7b模型进行微调。该模型在开源数据集[Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca)上进行微调，并使用[直接偏好优化（DPO）算法](https://arxiv.org/abs/2305.18290)进行对齐。更多详情请参见[model card](https://huggingface.co/Intel/neural-chat-7b-v3-1)和[博客](https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3)。  
* **notus-7b-v1** - Notus是一组使用[直接偏好优化（DPO）](https://arxiv.org/abs/2305.18290)和相关[RLHF](https://huggingface.co/blog/rlhf)技术微调的模型。此模型是第一个版本，基于zephyr-7b-sft使用DPO进行微调。采用数据优先的方法，Notus-7B-v1与Zephyr-7B-beta的唯一区别在于用于DPO的偏好数据集。所提出的数据集创建方法有助于有效微调Notus-7b，使其在[AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)中超越Zephyr-7B-beta和Claude 2。更多模型详情请参见[model card](https://huggingface.co/argilla/notus-7b-v1)。  
* **youri-7b-chat** - Youri-7b-chat是一个基于Llama2的模型。[Rinna Co., Ltd.](https://rinna.co.jp/)使用英语和日语数据集的混合对Llama2模型进行了进一步预训练，以提升日语任务能力。该模型在Hugging Face hub上公开发布。您可以在[rinna/youri-7b-chat项目页面](https://huggingface.co/rinna/youri-7b)找到详细信息。  
* **baichuan2-7b-chat** - 百川2是[百川智能](https://www.baichuan-ai.com/home)推出的最新一代大规模开源语言模型。它在2.6万亿个标记的高质量语料库上进行训练，在同规模的权威中文和英文基准测试中取得了最佳性能。  
* **internlm2-chat-1.8b** - InternLM2是InternLM系列的第二代。与上一代模型相比，它在推理、数学和编码等多方面能力上有了显著提升。更多模型详情请参见[model repository](https://huggingface.co/internlm)。  
* **glm-4-9b-chat** - GLM-4-9B是智谱AI推出的GLM-4系列最新一代预训练模型的开源版本。在语义、数学、推理、代码和知识数据集的评估中，GLM-4-9B及其对齐人类偏好的版本GLM-4-9B-Chat在性能上超越了Llama-3-8B。除了多轮对话，GLM-4-9B-Chat还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最多128K上下文）等高级功能。更多模型详情请参见[model card](https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/README_en.md)、[技术报告](https://arxiv.org/pdf/2406.12793)和[代码库](https://github.com/THUDM/GLM-4)。  
* **minicpm3-4b** - MiniCPM3-4B是MiniCPM系列的第三代。MiniCPM3-4B的整体性能超越了Phi-3.5-mini-Instruct，与许多近期的7B~9B模型相当。与前几代相比，MiniCPM3-4B具有更强大和更通用的技能集，以支持更广泛的应用。更多详情请参见[model card](https://huggingface.co/openbmb/MiniCPM3-4B)。  
* **DeepSeek-R1-Distill-Qwen-1.5B** - 使用[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)生成的推理数据对Qwen2.5-1.5B进行微调。您可以在[model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)中找到更多信息。  
* **DeepSeek-R1-Distill-Qwen-7B** - 使用[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)生成的推理数据对Qwen2.5-7B进行微调。您可以在[model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)中找到更多信息。  
* **DeepSeek-R1-Distill-Llama-8B** - 使用[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)生成的推理数据对Llama-3.1-8B进行微调。您可以在[model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)中找到更多信息。  
* **GLM-4-9B-0414** - GLM-4-32B-0414系列模型，拥有320亿个参数。其性能可与OpenAI的GPT系列和DeepSeek V3/R1系列相媲美。它还支持非常用户友好的本地部署功能。GLM-4-32B-Base-0414在15T的高质量数据上进行了预训练，包括大量推理型合成数据。您可以在[model card](https://huggingface.co/THUDM/GLM-4-9B-0414)中找到更多信息。  
* **GLM-Z1-32B-0414** - GLM-Z1-32B-0414是一个具有深度思考能力的推理模型。该模型基于GLM-4-32B-0414，通过冷启动、扩展强化学习和在数学、代码和逻辑等任务上的进一步训练开发而成。与基础模型相比，GLM-Z1-32B-0414在数学能力和解决复杂任务方面有显著提升。您可以在[model card](https://huggingface.co/THUDM/GLM-Z1-9B-0414)中找到更多信息。  
* **Qwen3-1.7/4B/8B/14B** - Qwen3是Qwen系列的最新一代大语言模型，提供了一套密集和专家混合（MoE）模型。在训练数据、模型架构和优化技术的广泛进步基础上，Qwen3在之前发布的Qwen2.5基础上实现了以下关键改进。您可以在[model card](https://huggingface.co/Qwen/Qwen3-8B)中找到更多信息。  
* **AFM-4.5B** - AFM-4.5B是由Arcee.ai开发的45亿参数指令调优模型，专为在从云到边缘的各种部署环境中提供企业级性能而设计。基础模型在8万亿个标记的数据集上进行训练，包括6.5万亿个通用预训练标记，以及1.5万亿个重点提升数学推理和代码生成能力的中期训练标记。预训练后，模型在高质量指令数据集上进行了监督微调。指令调优模型进一步通过可验证奖励的强化学习以及人类偏好进行优化。您可以在[model card](https://huggingface.co/arcee-ai/AFM-4.5B)中找到更多信息。

下图展示了提供的用户指令和模型回答示例。

![example](https://user-images.githubusercontent.com/29454499/255799218-611e7189-8979-4ef5-8a80-5a75e0136b50.png)

## 笔记本内容

本教程包括以下步骤：

- 安装先决条件
- 使用[OpenVINO与Hugging Face Optimum的集成](https://huggingface.co/blog/openvino)从公共源下载并转换模型
- 使用[NNCF](https://github.com/openvinotoolkit/nncf)将模型权重压缩为INT4或INT8精度
- 创建推理管道
- 运行聊天机器人/文档问答

## 安装说明
这是一个自包含的示例，仅依赖于其自身的代码。  
我们建议在虚拟环境中运行笔记本。您只需要一个Jupyter服务器即可开始。  
更多详情，请参见[安装指南](../../README.md)。  
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-chatbot/README.md" />