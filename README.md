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

* OpenChineseLLaMA：
  * 地址：https://github.com/OpenLMLab/OpenChineseLLaMA
  * 简介：基于 LLaMA-7B 经过中文数据集增量预训练产生的中文大语言模型基座，对比原版 LLaMA，该模型在中文理解能力和生成能力方面均获得较大提升，在众多下游任务中均取得了突出的成绩。

* BELLE：
  * 地址：https://github.com/LianjiaTech/BELLE
  * 简介：包括训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。BELLE针对中文做了优化

* Panda：
  * 地址：https://github.com/dandelionsllm/pandallm
  * 简介：模型目前基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练, 使用了接近 15M 条数据

* Luotuo-Chinese-LLM：
  * 地址：https://github.com/LC1332/Luotuo-Chinese-LLM
  * 简介：中文大语言模型开源项目，包含了一系列大规模语言模型，指令微调数据集等。

* Linly：
  * 地址：https://github.com/CVI-SZU/Linly
  * 简介：提供中文对话模型 Linly-ChatFlow 、中文基础模型 Linly-Chinese-LLaMA 及其训练数据。 中文基础模型以 LLaMA 为底座，利用中文和中英平行增量预训练。项目汇总了目前公开的多语言指令数据，对中文模型进行了大规模指令跟随训练，实现了 Linly-ChatFlow 对话模型。

* ChatYuan
  * 地址：https://github.com/clue-ai/ChatYuan
  * 简介：元语智能发布的一系列支持中英双语的功能型对话语言大模型，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

* CPM-Bee
  * 地址：https://github.com/OpenBMB/CPM-Bee
  * 简介：一个完全开源、允许商用的百亿参数中英文基座模型。它采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型。

###  2. <a name='Application'></a>Application

#### 2.1 垂直领域微调

##### 医疗

* DoctorGLM：
  * 地址：https://github.com/xionghonglin/DoctorGLM
  * 简介：基于 ChatGLM-6B的中文问诊模型，通过中文医疗对话数据集进行微调，实现了包括lora、p-tuningv2等微调及部署

* BenTsao：
  * 地址：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese
  * 简介：开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对LLaMA进行了指令微调，提高了LLaMA在医疗领域的问答效果。

* BianQue：
  * 地址：https://github.com/scutcyr/BianQue
  * 简介：一个经过指令与多轮问询对话联合微调的医疗对话大模型，基于ClueAI/ChatYuan-large-v2作为底座，使用中文医疗问答指令与多轮问询对话混合数据集进行微调。
  
* HuatuoGPT：
  * 地址：https://github.com/FreedomIntelligence/HuatuoGPT
  * 简介：开源了经过中文医学指令精调/指令微调(Instruct-tuning)的一个GPT-like模型

* Med-ChatGLM：
  * 地址：https://github.com/SCIR-HI/Med-ChatGLM
  * 简介：基于中文医学知识的ChatGLM模型微调，微调数据与BenTsao相同。

* QiZhenGPT：
  * 地址：https://github.com/CMKRG/QiZhenGPT
  * 简介：该项目利用启真医学知识库构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调，大幅提高了模型在中文医疗场景下效果，首先针对药品知识问答发布了评测数据集，后续计划优化疾病、手术、检验等方面的问答效果，并针对医患问答、病历自动生成等应用展开拓展。

* XrayGLM，首个会看胸部X光片的中文多模态医学大模型：
  * 地址：https://github.com/WangRongsheng/XrayGLM
  * 简介：该项目为促进中文领域医学多模态大模型的研究发展，发布了XrayGLM数据集及模型，其在医学影像诊断和多轮交互对话上显示出了非凡的潜力。


##### 法律

* LaWGPT：基于中文法律知识的大语言模型
  * 地址：https://github.com/pengxiao-song/LaWGPT
  * 简介：该系列模型在通用中文基座模型（如 Chinese-LLaMA、ChatGLM 等）的基础上扩充法律领域专有词表、大规模中文法律语料预训练，增强了大模型在法律领域的基础语义理解能力。在此基础上，构造法律领域对话问答数据集、中国司法考试数据集进行指令精调，提升了模型对法律内容的理解和执行能力。

* LexiLaw，中文法律大模型
  * 地址：https://github.com/CSHaitao/LexiLaw
  * 简介：LexiLaw 是一个基于 ChatGLM-6B微调的中文法律大模型，通过在法律领域的数据集上进行微调。该模型旨在为法律从业者、学生和普通用户提供准确、可靠的法律咨询服务，包括具体法律问题的咨询，还是对法律条款、案例解析、法规解读等方面的查询。

##### 金融

* Cornucopia（聚宝盆）：基于中文金融知识的LLaMA微调模型
  * 地址：https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese
  * 简介：开源了经过中文金融知识指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过中文金融公开数据+爬取的金融数据构建指令数据集，并在此基础上对LLaMA进行了指令微调，提高了 LLaMA 在金融领域的问答效果。基于相同的数据，后期还会利用GPT3.5 API构建高质量的数据集，另在中文知识图谱-金融上进一步扩充高质量的指令数据集。

* BBT-FinCUGE-Applications
  * 地址：https://github.com/ssymmetry/BBT-FinCUGE-Applications
  * 简介：开源了中文金融领域开源语料库BBT-FinCorpus，中文金融领域知识增强型预训练语言模型BBT-FinT5及中文金融领域自然语言处理评测基准CFLEB。

* XuanYuan（轩辕）：首个千亿级中文金融对话模型
  * 地址：https://huggingface.co/xyz-nlp/XuanYuan2.0
  * 简介：轩辕是国内首个开源的千亿级中文对话大模型，同时也是首个针对中文金融领域优化的千亿级开源对话大模型。轩辕在BLOOM-176B的基础上针对中文通用领域和金融领域进行了针对性的预训练与微调，它不仅可以应对通用领域的问题，也可以解答与金融相关的各类问题，为用户提供准确、全面的金融信息和建议。

##### 数学

* chatglm-maths：
  * 地址：https://github.com/yongzhuo/chatglm-maths
  * 简介：基于chatglm-6b微调/LORA/PPO/推理的数学题解题大模型, 样本为自动生成的整数/小数加减乘除运算, 可gpu/cpu部署，开源了训练数据集等。

##### 文化

* Firefly：
  * 地址：https://github.com/yangjianxin1/Firefly
  * 简介：中文对话式大语言模型，构造了许多与中华文化相关的数据，以提升模型这方面的表现，如对联、作诗、文言文翻译、散文、金庸小说等。

#### 2.2 LangChain应用

* Chinese-LangChain：
  * 地址：https://github.com/yanqiangmiffy/Chinese-LangChain
  * 简介：基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成（包括互联网检索结果接入）

* langchain-ChatGLM：
  * 地址：https://github.com/imClumsyPanda/langchain-ChatGLM
  * 简介：基于本地知识库的 ChatGLM 等大语言模型应用实现

#### 2.3 外部挂件应用

* wenda：
  * 地址：https://github.com/wenda-LLM/wenda
  * 简介：一个LLM调用平台。为小模型外挂知识库查找和设计自动执行动作，实现不亚于于大模型的生成能力。

* JittorLLMs：
  * 地址：https://github.com/Jittor/JittorLLMs
  * 简介：计图大模型推理库：笔记本没有显卡也能跑大模型，具有成本低，支持广，可移植，速度快等优势。

* WebCPM
  * 地址：https://github.com/thunlp/WebCPM
  * 简介：一个支持可交互网页搜索的中文大模型。 

* GPT Academic：
  * 地址：https://github.com/binary-husky/gpt_academic
  * 简介：为GPT/GLM提供图形交互界面，特别优化论文阅读润色体验，支持并行问询多种LLM模型，支持清华chatglm等本地模型。兼容复旦MOSS, llama, rwkv, 盘古等。

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

* Chinese medical dialogue data：
  * 地址：https://github.com/Toyhom/Chinese-medical-dialogue-data
  * 数据集说明：中文医疗对话数据集，包括：<Andriatria_男科> 94596个问答对 <IM_内科> 220606个问答对 <OAGD_妇产科> 183751个问答对 <Oncology_肿瘤科> 75553个问答对 <Pediatric_儿科> 101602个问答对 <Surgical_外科> 115991个问答对 总计 792099个问答对。


###  4. <a name='Evaluation'></a>Evaluation

* C-Eval: 构造中文大模型的知识评估基准：
  * 地址：https://yaofu.notion.site/C-Eval-6b79edd91b454e3d8ea41c59ea2af873
  * 简介：构造了一个覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集。此外还给出了当前主流中文LLM的评测结果。

* SuperCLUElyb: SuperCLUE琅琊榜
  * 地址：https://github.com/CLUEbenchmark/SuperCLUElyb
  * 简介：中文通用大模型匿名对战评价基准，这是一个中文通用大模型对战评价基准，它以众包的方式提供匿名、随机的对战。他们发布了初步的结果和基于Elo评级系统的排行榜。

* GAOKAO-Bench: 
  * 地址：https://github.com/OpenLMLab/GAOKAO-Bench
  * 简介：GAOKAO-bench是一个以中国高考题目为数据集，测评大模型语言理解能力、逻辑推理能力的测评框架，收集了2010-2022年全国高考卷的题目，其中包括1781道客观题和1030道主观题，构建起GAOKAO-bench的数据部分。

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

* DecryptPrompt：
  * 地址：https://github.com/DSXiangLi/DecryptPrompt
  * 简介：总结了Prompt&LLM论文，开源数据&模型，AIGC应用。


## 作者

* [HqWu-HITCS](https://github.com/HqWu-HITCS)
* [thinkwee](https://github.com/thinkwee)

## Star History

<a href="https://star-history.com/#HqWu-HITCS/Awesome-Chinese-LLM&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-Chinese-LLM&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-Chinese-LLM&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-Chinese-LLM&type=Date" />
  </picture>
</a>

