# 开源大语言模型完整列表

Large Language Model (LLM) 即大规模语言模型，是一种基于深度学习的自然语言处理模型，它能够学习到自然语言的语法和语义，从而可以生成人类可读的文本。

所谓"语言模型"，就是只用来处理语言文字（或者符号体系）的 AI 模型，发现其中的规律，可以根据提示 (prompt)，自动生成符合这些规律的内容。

LLM 通常基于神经网络模型，使用大规模的语料库进行训练，比如使用互联网上的海量文本数据。这些模型通常拥有数十亿到数万亿个参数，能够处理各种自然语言处理任务，如自然语言生成、文本分类、文本摘要、机器翻译、语音识别等。

本文对国内外公司、科研机构等组织开源的 LLM 进行了全面的整理。

![](https://oscimg.oschina.net/oscnet/up-185402e608c2ac1ec44224641ad4c76af09.png)

---

## 开源中文 LLM

### [ChatGLM-6B —— 双语对话语言模型](https://www.oschina.net/p/chatglm-6b)

ChatGLM-6B 是一个开源的、支持中英双语问答的对话语言模型，并针对中文进行了优化。该模型基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。

ChatGLM-6B 使用了和 ChatGLM 相同的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 虽然规模不及千亿模型，但大大降低了推理成本，提升了效率，并且已经能生成相当符合人类偏好的回答。

### [MOSS —— 支持中英双语的对话大语言模型](https://www.oschina.net/p/moss)

MOSS 是一个支持中英双语和多种插件的开源对话语言模型， `moss-moon` 系列模型具有 160 亿参数，在 FP16 精度下可在单张 A100/A800 或两张 3090 显卡运行，在 INT4/8 精度下可在单张 3090 显卡运行。

MOSS 基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。

### [伶荔 (Linly) —— 大规模中文语言模型](https://www.oschina.net/p/linly)

相比已有的中文开源模型，伶荔模型具有以下优势：

1.  在 32*A100 GPU 上训练了不同量级和功能的中文模型，对模型充分训练并提供强大的 baseline。据知，33B 的 Linly-Chinese-LLAMA 是目前最大的中文 LLaMA 模型。
    
2.  公开所有训练数据、代码、参数细节以及实验结果，确保项目的可复现性，用户可以选择合适的资源直接用于自己的流程中。
    
3.  项目具有高兼容性和易用性，提供可用于 CUDA 和 CPU 的量化推理框架，并支持 Huggingface 格式。
    

目前公开可用的模型有：

-   **Linly-Chinese-LLaMA**：中文基础模型，基于 LLaMA 在高质量中文语料上增量训练强化中文语言能力，现已开放 7B、13B 和 33B 量级，65B 正在训练中。
    
-   **Linly-ChatFlow**：中文对话模型，在 400 万指令数据集合上对中文基础模型指令精调，现已开放 7B、13B 对话模型。
    
-   **Linly-ChatFlow-int4** ：ChatFlow 4-bit 量化版本，用于在 CPU 上部署模型推理。
    

进行中的项目：

-   **Linly-Chinese-BLOOM**：基于 BLOOM 中文增量训练的中文基础模型，包含 7B 和 175B 模型量级，可用于商业场景。

### [Chinese-Vicuna —— 基于 LLaMA 的中文大语言模型](https://www.oschina.net/p/chinese-vicuna)

Chinese-Vicuna 是一个中文低资源的 LLaMA+Lora 方案。

项目包括

-   finetune 模型的代码
-   推理的代码
-   仅使用 CPU 推理的代码 (使用 C++)
-   下载 / 转换 / 量化 Facebook llama.ckpt 的工具
-   其他应用

### [Chinese-LLaMA-Alpaca —— 中文 LLaMA & Alpaca 大模型](https://www.oschina.net/p/chinese-llama-alpaca)

Chinese-LLaMA-Alpaca 包含中文 LLaMA 模型和经过指令微调的 Alpaca 大型模型。

这些模型在原始 LLaMA 的基础上，扩展了中文词汇表并使用中文数据进行二次预训练，从而进一步提高了对中文基本语义理解的能力。同时，中文 Alpaca 模型还进一步利用中文指令数据进行微调，明显提高了模型对指令理解和执行的能力。

### [ChatYuan —— 对话语言大模型](https://www.oschina.net/p/chatyuan)

ChatYuan 是一个支持中英双语的功能型对话语言大模型。ChatYuan-large-v2 使用了和 v1 版本相同的技术方案，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

ChatYuan-large-v2 是 ChatYuan 系列中以轻量化实现高质量效果的模型之一，用户可以在消费级显卡、 PC 甚至手机上进行推理（INT4 最低只需 400M ）。

### [华驼 (HuaTuo) —— 基于中文医学知识的 LLaMA 微调模型](https://www.oschina.net/p/huatuo-llama)

华驼 (HuaTuo) 是基于中文医学知识的 LLaMA 微调模型。

此项目开源了经过中文医学指令精调 / 指令微调 (Instruct-tuning) 的 LLaMA-7B 模型。通过医学知识图谱和 GPT3.5 API 构建了中文医学指令数据集，并在此基础上对 LLaMA 进行了指令微调，提高了 LLaMA 在医疗领域的问答效果。

### [鹏程·盘古α —— 中文预训练语言模型](https://www.oschina.net/p/pangu-alpha)

「鹏程·盘古α」是业界首个 2000 亿参数以中文为核心的预训练生成语言模型，目前开源了两个版本：鹏程·盘古α和鹏程·盘古α增强版，并支持NPU和GPU两个版本，支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备较强的少样本学习的能力。

基于盘古系列大模型提供大模型应用落地技术帮助用户高效的落地超大预训练模型到实际场景。整个框架特点如下：

![up-0721578aee3f791d625b711918c65f49b61.png](https://oscimg.oschina.net/oscnet/up-0721578aee3f791d625b711918c65f49b61.png)

主要有如下几个核心模块：

-   数据集：从开源开放数据集、common crawl 数据集、电子书等收集近 80TB 原始语料，构建了约 1.1TB 的高质量中文语料数据集、53 种语种高质量单、双语数据集 2TB。
    
-   基础模块：提供预训练模型库，支持常用的中文预训练模型，包括鹏程・盘古 α、鹏程・盘古 α 增强版等。
    
-   应用层：支持常见的 NLP 应用比如多语言翻译、开放域对话等，支持预训练模型落地工具，包括模型压缩、框架移植、可持续学习，助力大模型快速落地。
    

### [鹏程·盘古对话生成大模型](https://www.oschina.net/p/pangu-dialog)

鹏程・盘古对话生成大模型 (PanGu-Dialog)。

PanGu-Dialog 是以大数据和大模型为显著特征的大规模开放域对话生成模型，充分利用大规模预训练语言模型的知识和语言能力，构建可控、可靠可信、有智慧的自然人机对话模型。主要特性如下：

-   首次提出对话智慧度以探索对话模型的逻辑推理、数据计算、联想、创作等方面的能力。
-   构建了覆盖领域最广 (据我们所知) 的开放域交互式对话评估数据集 PGCED，12 个领域，并在知识性、安全性、智慧程度等方面制作了针对性的评测数据。
-   基于预训练 \+ 持续微调的学习策略融合大规模普通文本和多种对话数据训练而成，充分利用训练语言模型语言能力和知识，高效构建强大的对话模型。
-   在各项指标上达到了中文纯模型生成式对话 SOTA 水平，在知识性和信息量方面优势明显，但安全性、可靠、可信、可控、智慧等方面的提升并不明显。
-   目前生成式对话仍处于较低水平，与人类对话能力存在明显的差距，后续将在现有基础上针对不同的维度不断优化迭代，不断进步。

### [悟道 —— 双语多模态大语言模型](https://www.oschina.net/p/wudao-model)

“悟道” 是双语多模态预训练模型，规模达到 1.75 万亿参数。项目现有 7 个开源模型成果。

![](https://oscimg.oschina.net/oscnet/up-ea1775cd4c5a97421ce4ee9770f0460ce7d.png)

#### 图文类

-   **[CogView](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/CogView)**
    
    CogView 参数量为 40 亿，模型可实现文本生成图像，经过微调后可实现国画、油画、水彩画、轮廓画等图像生成。目前在公认 MS COCO 文生图任务上取得了超过 OpenAI DALL・E 的成绩，获得世界第一。
    
-   **[BriVL](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/BriVL)**
    
    BriVL (Bridging Vision and Language Model) 是首个中文通用图文多模态大规模预训练模型。BriVL 模型在图文检索任务上有着优异的效果，超过了同期其他常见的多模态预训练模型（例如 UNITER、CLIP）。
    

#### 文本类

-   **[GLM](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/GLM)**
    
    GLM 是以英文为核心的预训练语言模型系列，基于新的预训练范式实现单一模型在语言理解和生成任务方面取得了最佳结果，并且超过了在相同数据量进行训练的常见预训练模型（例如 BERT，RoBERTa 和 T5），目前已开源 1.1 亿、3.35 亿、4.10 亿、5.15 亿、100 亿参数规模的模型。
    
-   **[CPM](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/CPM)**
    
    CPM 系列模型是兼顾理解与生成能力的预训练语言模型系列，涵盖中文、中英双语多类模型，目前已开源 26 亿、110 亿和 1980 亿参数规模的模型。
    
-   **[Transformer-XL](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/Transformer-XL)**
    
    Transformer-XL 是以中文为核心的预训练语言生成模型，参数规模为 29 亿，目前可支持包括文章生成、智能作诗、评论 / 摘要生成等主流 NLG 任务。
    
-   **[EVA](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/EVA)**
    
    EVA 是一个开放领域的中文对话预训练模型，是目前最大的汉语对话模型，参数量达到 28 亿，并且在包括不同领域 14 亿汉语的悟道对话数据集（WDC）上进行预训练。
    
-   **[Lawformer](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/Lawformer)**
    
    Lawformer 是世界首创法律领域长文本中文预训练模型，参数规模达到 1 亿。
    

#### 蛋白质类

-   **[ProtTrans](https://git.openi.org.cn/BAAI/WuDao-Model/src/branch/master/ProtTrans)**
    
    ProtTrans 是国内最大的蛋白质预训练模型，参数总量达到 30 亿。

### [BBT-2 —— 120 亿参数大语言模型](https://www.oschina.net/p/bbt-2)

BBT-2 是包含 120 亿参数的通用大语言模型，在 BBT-2 的基础上训练出了代码，金融，文生图等专业模型。基于 BBT-2 的系列模型包括：

-   BBT-2-12B-Text：120 亿参数的中文基础模型
    
-   BBT-2.5-13B-Text: 130 亿参数的中文+英文双语基础模型
    
-   BBT-2-12B-TC-001-SFT 经过指令微调的代码模型，可以进行对话
    
-   BBT-2-12B-TF-001 在 120 亿模型上训练的金融模型，用于解决金融领域任务
    
-   BBT-2-12B-Fig：文生图模型
    
-   BBT-2-12B-Science 科学论文模型
   
### [BELLE —— 开源中文对话大模型](https://www.oschina.net/p/belle)

BELLE: Be Everyone's Large Language model Engine（开源中文对话大模型）

本项目目标是促进中文对话大模型开源社区的发展，愿景做能帮到每一个人的 LLM Engine。现阶段本项目基于一些开源预训练大语言模型（如 BLOOM），针对中文做了优化，模型调优仅使用由 ChatGPT 生产的数据（不包含任何其他数据）。

## 开源 LLM

### [LLaMA —— Meta 大语言模型](https://www.oschina.net/p/llama)

LLaMA 语言模型全称为 "Large Language Model Meta AI"，是 Meta 的全新大型语言模型（LLM），这是一个模型系列，根据参数规模进行了划分（分为 70 亿、130 亿、330 亿和 650 亿参数不等）。

其中 LaMA-13B（130 亿参数的模型）尽管模型参数相比 OpenAI 的 GPT-3（1750 亿参数） 要少了十几倍，但在性能上反而可以超过 GPT-3 模型。更小的模型也意味着开发者可以在 PC 甚至是智能手机等设备上本地运行类 ChatGPT 这样的 AI 助手，无需依赖数据中心这样的大规模设施。

### [Stanford Alpaca —— 指令调优的 LLaMA 模型](https://www.oschina.net/p/stanford-alpaca)

Stanford Alpaca（斯坦福 Alpaca）是一个指令调优的 LLaMA 模型，从 Meta 的大语言模型 LLaMA 7B 微调而来。

Stanford Alpaca 让 OpenAI 的 text-davinci-003 模型以 self-instruct 方式生成 52K 指令遵循（instruction-following）样本，以此作为 Alpaca 的训练数据。研究团队已将训练数据、生成训练数据的代码和超参数开源，后续还将发布模型权重和训练代码。

### [Lit-LLaMA —— 基于 nanoGPT 的语言模型](https://www.oschina.net/p/lit-llama)

Lit-LLaMA 是一个基于 nanoGPT 的 LLaMA 语言模型的实现，支持量化、LoRA 微调、预训练、flash attention、LLaMA-Adapter 微调、Int8 和 GPTQ 4bit 量化。

主要特点：单一文件实现，没有样板代码；在消费者硬件上或大规模运行；在数值上等同于原始模型。

Lit-LLaMA 认为人工智能应该完全开源并成为集体知识的一部分。但原始的 LLaMA 代码采用 GPL 许可证，这意味着使用它的任何项目也必须在 GPL 下发布。这“污染”了其他代码，阻止了与生态系统的集成。Lit-LLaMA 永久性地解决了这个问题。

### [GloVe —— 斯坦福大学的词向量工具](https://nlp.stanford.edu/projects/glove/)

GloVe的全称叫Global Vectors for Word Representation，它是一个基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

以下是 GloVe 提供的预训练词向量，遵循 [Public Domain Dedication and License](http://opendatacommons.org/licenses/pddl/) 许可。

-   [Wikipedia 2014](http://dumps.wikimedia.org/enwiki/20140102/)+[Gigaword 5](https://catalog.ldc.upenn.edu/LDC2011T07)(6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download):[glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)
-   Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download):[glove.42B.300d.zip](https://nlp.stanford.edu/data/glove.42B.300d.zip)
-   Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):[glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)
-   Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download):[glove.twitter.27B.zip](https://nlp.stanford.edu/data/glove.twitter.27B.zip)

### [Dolly —— 低成本大语言模型](https://www.oschina.net/p/dolly)

Dolly 是一个低成本的 LLM，Dolly 采用 EleutherAI 现有的 60 亿参数的开源模型，并对其进行细微的修改，以激发指令跟随能力。

尽管模型小得多，只有 60 亿个参数，以及较小的数据集和训练时间（ChatGPT 的参数是 1750 亿个），但 Dolly 仍然表现出了 ChatGPT 所展示的同样的 "神奇的人类互动能力"。


### [OPT-175B —— Meta 开源的大语言模型](https://www.oschina.net/p/opt-175b)

OPT-175B 是 Meta 开源的大语言模型，拥有超过 1750 亿个参数 —— 和 GPT-3 相当。相比 GPT-3，OPT-175B 的优势在于它完全免费。

Meta 还公布了代码库、开发过程日志、数据、研究论文和其他与 OPT-175B 相关的信息。尽管 OPT-175B 是免费的，但 Meta 也给出了一些限制。为了防止误用和 “保持完整性”，OPT-175B 只允许在非商业用途下使用。也就是说，OPT-175B 的多数应用场景还是在科研上。

### [Cerebras-GPT —— 自然语言处理领域大模型](https://www.oschina.net/p/cerebras-gpt)

Cerebras GPT 是由 Cerebras 公司开源的自然语言处理领域的预训练大模型，其模型参数规模最小 1.11 亿，最大 130 亿，共 7 个模型。

与业界的模型相比，Cerebras-GPT 几乎是各个方面完全公开，没有任何限制。不管是模型架构，还是预训练结果都是公开的。

### [BLOOM —— 自然语言处理大模型](https://www.oschina.net/p/bloom)

Bloom 是用于自然语言处理的大语言模型，包含 1760 亿个参数，支持 46 种自然语言（包括中文）和 13 种编程语言，可以用来回答问题、翻译文本、从文件中提取信息片段，还能像 GitHub Copilot 一样用于生成代码。

BLOOM 模型的最大优势是它的易获取性，任何个人或机构都可以从 Hugging Face 免费获得 1760 亿个参数的完整模型。用户有多个语种可选，然后将需求输入到 BLOOM 中，任务类型包括撰写食谱或诗歌、翻译或总结文本，甚至还有代码编程。人工智能开发者可以在该模型的基础上构建他们自己的应用程序。

### [GPT-J —— 自然语言处理 AI 模型](https://www.oschina.net/p/gpt-j)

GPT-J 是一个基于 GPT-3，由 60 亿个参数组成的自然语言处理 AI 模型。

该模型在一个 800GB 的开源文本数据集上进行训练，并且能够与类似规模的 GPT-3 模型相媲美。 该模型通过利用 Google Cloud 的 v3-256 TPU 以及 EleutherAI 的 The Pile 数据集进行训练，历时大约五周时间。GPT-J 在标准 NLP 基准工作负载上实现了与 OpenAI 报告的 67 亿参数版本的 GPT-3 类似的准确性。模型代码、预训练的权重文件、Colab 文档和一个演示网页都包含在 EleutherAI 的开源项目中。

### [GPT-2 —— 基于 Transformer 的大型语言模型](https://www.oschina.net/p/gpt-2)

GPT-2 是一种基于 transformer 的大型语言模型，具有 15 亿个参数，在 800 万网页数据集上进行训练。

GPT-2 能够翻译文本、回答问题、总结段落，并生成文本输出。虽然其输出内容有时与人类相似，但在生成长段落时输出内容可能会变得重复或无意义。

GPT-2 是一个通用学习器，没有经过专门训练来执行任何特定的任务，并且是作为 OpenAI 2018 GPT 模型的“直接扩展”而创建的，其参数数量和训练数据集的大小均增加了十倍。

### [RWKV-LM —— 线性 Transformer 模型](https://www.oschina.net/p/rwkv)

RWKV 是结合了 RNN 和 Transformer 的语言模型，适合长文本，运行速度较快，拟合性能较好，占用显存较少，训练用时较少。

RWKV 整体结构依然采用 Transformer Block 的思路，相较于原始 Transformer Block 的结构，RWKV 将 self-attention 替换为 Position Encoding 和 TimeMix，将 FFN 替换为 ChannelMix。其余部分与 Transfomer 一致。    

### [白泽 —— 使用 LoRA 训练的大语言模型](https://www.oschina.net/p/baize-chatbot)

白泽是使用 LoRA 训练的开源聊天模型，它改进了开源大型语言模型 LLaMA，通过使用新生成的聊天语料库对 LLaMA 进行微调，该模型在单个 GPU 上运行，使其可供更广泛的研究人员使用。

白泽目前包括四种英语模型：白泽 -7B、13B 和 30B（通用对话模型），以及一个垂直领域的白泽 - 医疗模型，供研究 / 非商业用途使用，并计划在未来发布中文的白泽模型。

白泽的数据处理、训练模型、Demo 等全部代码已经开源。

### [**CodeGeeX** —— 多语言代码生成模型](https://www.oschina.net/p/codegeex)

CodeGeeX 是一个具有 130 亿参数的多编程语言代码生成预训练模型。CodeGeeX 采用华为 MindSpore 框架实现，在鹏城实验室 “鹏城云脑 II” 中的 192 个节点（共 1536 个国产昇腾 910 AI 处理器）上训练而成。

CodeGeeX 有以下特点：

-   高精度代码生成：支持生成 Python、C++、Java、JavaScript 和 Go 等多种主流编程语言的代码，在 HumanEval-X 代码生成任务上取得 47%~60% 求解率，较其他开源基线模型有更佳的平均性能。
-   跨语言代码翻译：支持代码片段在不同编程语言间进行自动翻译转换，翻译结果正确率高，在 HumanEval-X 代码翻译任务上超越了其它基线模型。
-   自动编程插件：CodeGeeX 插件现已上架 VSCode 插件市场（完全免费），用户可以通过其强大的少样本生成能力，自定义代码生成风格和能力，更好辅助代码编写。
-   模型跨平台开源: 所有代码和模型权重开源开放，用作研究用途。CodeGeeX 同时支持昇腾和英伟达平台，可在单张昇腾 910 或英伟达 V100/A100 上实现推理。

### [Vicuna —— 基于 LLaMA 的微调大语言模型](https://www.oschina.net/p/vicuna)

Vicuna 模型对 LLaMA 进行了微调，由加州大学伯克利分校、卡内基梅隆大学、斯坦福大学、加州大学圣地亚哥分校和 MBZUAI 的学术团队进行微调训练而成，有两种大小可供选择：7B 和 13B。

Vicuna-13B 与 Stanford Alpaca 等其他开源模型相比展示了具有竞争力的性能。

以 GPT-4 为评判标准的初步评估显示，Vicuna-13B 达到了 OpenAI ChatGPT 和 Google Bard 90% 以上的质量，同时在 90% 以上的情况下超过了 LLaMA 和 Stanford Alpaca 等其他模型的表现。训练 Vicuna-13B 成本约为 300 美元。训练和服务代码，以及在线演示都是公开的，可用于非商业用途。

### [RedPajama —— 1.2 万亿数据集的可商用大语言模型](https://www.oschina.net/p/redpajama)

RedPajama 项目旨在创建一套领先的全开源大语言模型。目前，该项目已完成了第一步，成功复制了 LLaMA 训练数据集超过 1.2 万亿个数据 token。该项目由 Together、[Ontocord.ai](http://Ontocord.ai)、ETH DS3Lab、斯坦福大学 CRFM、Hazy Research 和 MILA 魁北克 AI 研究所联合开发。

RedPajama 包含三个主要组成部分：预训练数据、基础模型和指令调优数据与模型。

### [OpenAssistant —— 基于对话的大型语言模型](https://www.oschina.net/p/open-assistant)

OpenAssistant 是一个开源项目，旨在开发免费提供给所有人使用的 AI 聊天机器人。

训练数据集 OpenAssistant Conversations 包含了超过 60 万个涉及各种主题的交互，用于训练各种模型。目前发布了经过指令调整的 LLaMA 13B 和 30B 模型，以及其他使用相同数据集训练的模型。

### [StableLM —— Stability AI 开发的语言模型](https://www.oschina.net/p/stablelm)

StableLM 项目仓库包含 Stability AI 正在进行的 StableLM 系列语言模型开发，目前 Stability AI 发布了初始的 StableLM-alpha 模型集，具有 30 亿和 70 亿参数。150 亿和 300 亿参数的模型正在开发中。

StableLM 模型可以生成文本和代码，并为一系列下游应用提供支持。它们展示了小而高效的模型如何在适当的训练下提供高性能。

### [StarCoder —— AI 编程模型](https://www.oschina.net/p/starcoder)

StarCoder（150 亿参数）是 Hugging Face 联合 ServiceNow 发布的免费大型语言模型，该模型经过训练主要用途是可以生成代码，目的是为了对抗 GitHub Copilot 和亚马逊 CodeWhisperer 等基于 AI 的编程工具。

### [SantaCoder —— 轻量级 AI 编程模型](https://www.oschina.net/p/santacoder)

SantaCoder 是一个语言模型，该模型拥有 11 亿个参数，可以用于 Python、Java 和 JavaScript 这几种编程语言的代码生成和补全建议。

根据官方提供的信息，训练 SantaCoder 的基础是 The Stack（v1.1）数据集，SantaCoder 虽然规模相对较小，只有 11 亿个参数，在参数的绝对数量上低于 InCoder（67 亿）或 CodeGen-multi（27 亿），但 SantaCoder 的表现则是要远好于这些大型多语言模型。

### [MLC LLM —— 本地大语言模型](https://www.oschina.net/p/mlc-llm)

MLC LLM 是一种通用解决方案，它允许将任何语言模型本地部署在各种硬件后端和本地应用程序上。

此外，MLC LLM 还提供了一个高效的框架，供使用者根据需求进一步优化模型性能。MLC LLM 旨在让每个人都能在个人设备上本地开发、优化和部署 AI 模型，而无需服务器支持，并通过手机和笔记本电脑上的消费级 GPU 进行加速。

### [Web LLM —— 浏览器大语言模型](https://www.oschina.net/p/web-llm)

Web LLM 是一个可将大型语言模型和基于 LLM 的聊天机器人引入 Web 浏览器的项目。一切都在浏览器内运行，无需服务器支持，并使用 WebGPU 加速。这开辟了许多有趣的机会，可以为每个人构建 AI 助手，并在享受 GPU 加速的同时实现隐私。

### [WizardLM —— 基于 LLaMA 的微调大语言模型](https://www.oschina.net/p/wizardlm)

WizardLM 是一个经过微调的 7B LLaMA 模型。它通过大量具有不同难度的指令跟随对话进行微调。这个模型的新颖之处在于使用了 LLM 来自动生成训练数据。

WizardLM 模型使用一种名为 Evol-Instruct（是一种使用 LLM 代人类自主批生成各种难度等级和技术范围的开放指令，以提高 LLM 能力的新方法）的新方法，通过 70k 个计算机生成的指令进行训练，该方法生成具有不同难度级别的指令。

### [YaLM 100B —— 千亿参数预训练语言模型](https://www.oschina.net/p/yalm-100b)

YaLM 100B是一个类似 GPT 的神经网络，用于生成和处理文本。

该模型利用了 1000 亿个参数，在 800 个 A100 显卡和 1.7 TB 在线文本、书籍以及海量其他英文和俄文资源的集群上训练该模型花了 65 天时间。

### [OpenLLaMA —— LLaMA 大语言模型的开源复现版本](https://www.oschina.net/p/openllama)

OpenLLaMA 是 Meta AI 的 LLaMA 大语言模型的开源复现版本，采用宽松许可证。

仓库包含经过训练的 2000 亿标记的 7B OpenLLaMA 模型的公共预览版，并提供了预训练的 OpenLLaMA 模型的 PyTorch 和 Jax 权重，以及评估结果和与原始 LLaMA 模型的比较。
