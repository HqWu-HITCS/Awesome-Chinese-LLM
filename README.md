# Awesome-Chinese-LLM

自ChatGPT为代表的大语言模型（Large Language Model, LLM）出现以后，由于其惊人的类通用人工智能（AGI）的能力，掀起了新一轮自然语言处理领域的研究和应用的浪潮。尤其是以ChatGLM、LLaMA等平民玩家都能跑起来的较小规模的LLM开源之后，业界涌现了非常多基于LLM的二次微调或应用的案例。本项目旨在收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料。

## 目录
  - [1. Model](#1-model)
  - [2. Application](#2-application)
  - [3. Dataset](#3-dataset)
  - [4. Evaluation](#4-evaluation)
  - [5. Tutorial](#5-tutorial)
  - [6. Related Repository](#6-related-repository)


###  1. <a name='Model'></a>Model

* ChatGLM：
  * 地址：https://github.com/THUDM/ChatGLM-6B
  * 简介：中文领域效果最好的开源底座模型之一，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持

* Moss：
  * 地址：https://github.com/OpenLMLab/MOSS
  * 简介：支持中英双语和多种插件的开源对话语言模型，MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。

* Chinese-LLaMA-Alpaca：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca
  * 简介：中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署，在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练

* BELLE：
  * 地址：https://github.com/LianjiaTech/BELLE
  * 简介：包括训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。BELLE针对中文做了优化

* Panda：
  * 地址：https://github.com/dandelionsllm/pandallm
  * 简介：模型目前基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练, 使用了接近 15M 条数据

* Luotuo-Chinese-LLM：
  * 地址：https://github.com/LC1332/Luotuo-Chinese-LLM
  * 简介：中文大语言模型开源项目，包含了一系列语言模型

* Linly：
  * 地址：https://github.com/CVI-SZU/Linly
  * 简介：提供中文对话模型 Linly-ChatFlow 、中文基础模型 Linly-Chinese-LLaMA 及其训练数据。 中文基础模型以 LLaMA 为底座，利用中文和中英平行增量预训练。项目汇总了目前公开的多语言指令数据，对中文模型进行了大规模指令跟随训练，实现了 Linly-ChatFlow 对话模型。

* ChatYuan
  * 地址：https://github.com/clue-ai/ChatYuan
  * 简介：元语智能发布的一个支持中英双语的功能型对话语言大模型


###  2. <a name='Application'></a>Application

* Chinese-LangChain：
  * 地址：https://github.com/yanqiangmiffy/Chinese-LangChain
  * 简介：基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成（包括互联网检索结果接入）

* langchain-ChatGLM：
  * 地址：https://github.com/imClumsyPanda/langchain-ChatGLM
  * 简介：基于本地知识库的 ChatGLM 等大语言模型应用实现

* DoctorGLM：
  * 地址：https://github.com/xionghonglin/DoctorGLM
  * 简介：基于 ChatGLM-6B的中文问诊模型，包括lora、p-tuningv2等微调及部署

* BenTsao：
  * 地址：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese
  * 简介：基于中文医学知识的LLaMA微调模型

* Med-ChatGLM：
  * 地址：https://github.com/SCIR-HI/Med-ChatGLM
  * 简介：基于中文医学知识的ChatGLM模型微调

* GPT Academic：
  * 地址：https://github.com/binary-husky/gpt_academic
  * 简介：为GPT/GLM提供图形交互界面，特别优化论文阅读润色体验，支持并行问询多种LLM模型，支持清华chatglm等本地模型。兼容复旦MOSS, llama, rwkv, 盘古等。

* wenda：
  * 地址：https://github.com/wenda-LLM/wenda
  * 简介：一个LLM调用平台。为小模型外挂知识库查找和设计自动执行动作，实现不亚于于大模型的生成能力。

* LaWGPT：
  * 地址：https://github.com/pengxiao-song/LaWGPT
  * 简介：基于中文法律知识的大语言模型

* Firefly：
  * 地址：https://github.com/yangjianxin1/Firefly
  * 简介：中文对话式大语言模型，构造了许多与中华文化相关的数据，以提升模型这方面的表现，如对联、作诗、文言文翻译、散文、金庸小说等。

* JittorLLMs：
  * 地址：https://github.com/Jittor/JittorLLMs
  * 简介：计图大模型推理库：笔记本没有显卡也能跑大模型，具有成本低，支持广，可移植，速度快等优势。

* WebCPM
  * 地址：https://github.com/thunlp/WebCPM
  * 简介：一个支持可交互网页搜索的中文大模型。 


###  3. <a name='Dataset'></a>Dataset

* alpaca_chinese_dataset：
  * 地址：https://github.com/hikariming/alpaca_chinese_dataset
  * 数据集说明：alpaca数据集进行中文翻译，并再制造一些对话数据

* Alpaca-CoT：
  * 地址：https://github.com/PhoebusSi/Alpaca-CoT
  * 数据集说明：统一了丰富的IFT数据（如CoT数据，目前仍不断扩充）、多种训练效率方法（如lora，p-tuning）以及多种LLMs，三个层面上的接口，打造方便研究人员上手的LLM-IFT研究平台。

* pCLUE：
  * 地址：https://github.com/CLUEbenchmark/pCLUE
  * 数据集说明：120万训练数据，73个Prompt，共包括9个任务

* firefly-train-1.1M：
  * 地址：https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
  * 数据集说明：23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万

* BELLE-data-1.5M：
  * 地址：https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M
  * 数据集说明：通过self-instruct生成，使用了中文种子任务，以及openai的text-davinci-003接口,涉及175个种子任务

* Chinese Scientific Literature Dataset：
  * 地址：https://github.com/ydli-ai/csl
  * 数据集说明：中文科学文献数据集（CSL），包含 396,209 篇中文核心期刊论文元信息 （标题、摘要、关键词、学科、门类）以及简单的prompt

###  4. <a name='Evaluation'></a>Evaluation

* C-Eval: 构造中文大模型的知识评估基准：
  * 地址：https://yaofu.notion.site/C-Eval-6b79edd91b454e3d8ea41c59ea2af873
  * 简介：构造了一个覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集。此外还给出了当前主流中文LLM的评测结果。

* PromptCBLUE: 中文医疗场景的LLM评测基准
  * 地址：https://github.com/michael-wzhu/PromptCBLUE
  * 简介：为推动LLM在医疗领域的发展和落地，由华东师范大学联合阿里巴巴天池平台，复旦大学附属华山医院，东北大学，哈尔滨工业大学（深圳），鹏城实验室与同济大学推出PromptCBLUE评测基准, 将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务,形成首个中文医疗场景的LLM评测基准。

* Open LLM Leaderboard：
  * 地址：https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
  * 简介：由HuggingFace组织的一个LLM评测榜单，目前已评估了较多主流的开源LLM模型。评估主要包括AI2 Reasoning Challenge, HellaSwag, MMLU, TruthfulQA四个数据集上的表现。

###  5. <a name='Tutorial'></a>Tutorial

* LangChain 🦜️🔗 中文网，跟着LangChain一起学LLM/GPT开发：
  * 地址：https://github.com/ydli-ai/csl
  * 简介：Langchain的中文文档，由是两个在LLM创业者维护，希望帮助到从刚进入AI应用开发的朋友们。

* LLMs九层妖塔：
  * 地址：https://github.com/km1994/LLMsNineStoryDemonTower
  * 简介：ChatGLM、Chinese-LLaMA-Alpaca、MiniGPT-4、FastChat、LLaMA、gpt4all等实战与经验。
 
 ###  6. <a name='Related Repository'></a>Related Repository
 
* FindTheChatGPTer：
  * 地址：https://github.com/chenking2020/FindTheChatGPTer
  * 简介：ChatGPT爆火，开启了通往AGI的关键一步，本项目旨在汇总那些ChatGPT的开源平替们，包括文本大模型、多模态大模型等，为大家提供一些便利。

* LLM_reviewer：
  * 地址：https://github.com/SpartanBin/LLM_reviewer
  * 简介：总结归纳近期井喷式发展的大语言模型，以开源、规模较小、可私有化部署、训练成本较低的‘小羊驼类’模型为主。

* Awesome-AITools：
  * 地址：https://github.com/ikaijua/Awesome-AITools/blob/main/README-CN.md
  * 简介：收藏整理了AI相关的实用工具、评测和相关文章。


## 作者

* [HqWu-HITCS](https://github.com/HqWu-HITCS)
* [thinkwee](https://github.com/thinkwee)
