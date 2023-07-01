<h1 align="center">
Awesome-Chinese-LLM
</h1>
<p align="center" width="100%">
<img src="src/icon.png" alt="Awesome-Chinese-LLM" style="width: 20%; height: auto; display: inline-block; margin: auto; border-radius: 50%;">
</p>
<p align="center">
<font face="黑体" color=orange size=5"> An Awesome Collection for LLM in Chinese </font>
</p>
<p align="center">
<font face="黑体" color=orange size=5"> 收集和梳理中文LLM相关 </font>
</p>
<p align="center">
  <a href="https://github.com/HqWu-HITCS/Awesome-Chinese-LLM/stargazers"> <img src="https://img.shields.io/github/stars/HqWu-HITCS/Awesome-Chinese-LLM.svg?style=popout-square" alt="GitHub stars"></a>
  <a href="https://github.com/HqWu-HITCS/Awesome-Chinese-LLM/issues"> <img src="https://img.shields.io/github/issues/HqWu-HITCS/Awesome-Chinese-LLM.svg?style=popout-square" alt="GitHub issues"></a>
  <a href="https://github.com/HqWu-HITCS/Awesome-Chinese-LLM/forks"> <img src="https://img.shields.io/github/forks/HqWu-HITCS/Awesome-Chinese-LLM.svg?style=popout-square" alt="GitHub forks"></a>
</p>


自ChatGPT为代表的大语言模型（Large Language Model, LLM）出现以后，由于其惊人的类通用人工智能（AGI）的能力，掀起了新一轮自然语言处理领域的研究和应用的浪潮。尤其是以ChatGLM、LLaMA等平民玩家都能跑起来的较小规模的LLM开源之后，业界涌现了非常多基于LLM的二次微调或应用的案例。本项目旨在收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料。

如果本项目能给您带来一点点帮助，麻烦点个⭐️吧～

同时也欢迎大家贡献本项目未收录的开源模型、应用、数据集等。提供新的仓库信息请发起PR，并按照本项目的格式提供仓库链接、star数，简介等相关信息，感谢~

<p align="center" width="100%">
<img src="src/chinese_taxonomy.png" alt="Awesome-Chinese-LLM">
</p>

## 目录
- [目录](#目录)
  - [1. Model](#1-model)
  - [2. Application](#2-application)
    - [2.1 垂直领域微调](#21-垂直领域微调)
      - [医疗](#医疗)
      - [法律](#法律)
      - [金融](#金融)
      - [教育](#教育)
      - [数学](#数学)
      - [文化](#文化)
    - [2.2 LangChain应用](#22-langchain应用)
    - [2.3 外部挂件应用](#23-外部挂件应用)
  - [3. Dataset](#3-dataset)
  - [4. Evaluation](#4-evaluation)
  - [5. Tutorial](#5-tutorial)
  - [6. Related Repository](#6-related-repository)
- [作者](#作者)
- [Star History](#star-history)


###  1. <a name='Model'></a>Model

* ChatGLM：
  * 地址：https://github.com/THUDM/ChatGLM-6B
![](https://img.shields.io/github/stars/THUDM/ChatGLM-6B.svg)
  * 简介：中文领域效果最好的开源底座模型之一，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持

* VisualGLM-6B
  * 地址：https://github.com/THUDM/VisualGLM-6B
![](https://img.shields.io/github/stars/THUDM/VisualGLM-6B.svg)
  * 简介：一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练。

* ChatGLM2-6B
  * 地址：https://github.com/THUDM/ChatGLM2-6B
![](https://img.shields.io/github/stars/THUDM/ChatGLM2-6B.svg)
  * 简介：基于开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，引入了GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练；基座模型的上下文长度扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练；基于 Multi-Query Attention 技术实现更高效的推理速度和更低的显存占用；允许商业使用。

* Chinese-LLaMA-Alpaca：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca
![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca.svg)
  * 简介：中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署，在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练

* OpenChineseLLaMA：
  * 地址：https://github.com/OpenLMLab/OpenChineseLLaMA
![](https://img.shields.io/github/stars/OpenLMLab/OpenChineseLLaMA.svg)
  * 简介：基于 LLaMA-7B 经过中文数据集增量预训练产生的中文大语言模型基座，对比原版 LLaMA，该模型在中文理解能力和生成能力方面均获得较大提升，在众多下游任务中均取得了突出的成绩。

* BELLE：
  * 地址：https://github.com/LianjiaTech/BELLE
![](https://img.shields.io/github/stars/LianjiaTech/BELLE.svg)
  * 简介：开源了基于BLOOMZ和LLaMA优化后的一系列模型，同时包括训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。

* Panda：
  * 地址：https://github.com/dandelionsllm/pandallm
![](https://img.shields.io/github/stars/dandelionsllm/pandallm.svg)
  * 简介：开源了基于LLaMA-7B, -13B, -33B, -65B 进行中文领域上的持续预训练的语言模型, 使用了接近 15M 条数据进行二次预训练。

* Fengshenbang-LM：
  * 地址：https://github.com/IDEA-CCNL/Fengshenbang-LM
![](https://img.shields.io/github/stars/IDEA-CCNL/Fengshenbang-LM.svg)
  * 简介：Fengshenbang-LM(封神榜大模型)是IDEA研究院认知计算与自然语言研究中心主导的大模型开源体系，该项目开源了姜子牙通用大模型V1，是基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。除姜子牙系列模型之外，该项目还开源了太乙、二郎神系列等模型。

* BiLLa：
  * 地址：https://github.com/Neutralzz/BiLLa
![](https://img.shields.io/github/stars/Neutralzz/BiLLa.svg)
  * 简介：该项目开源了推理能力增强的中英双语LLaMA模型。模型的主要特性有：较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；全量参数更新，追求更好的生成效果。

* Moss：
  * 地址：https://github.com/OpenLMLab/MOSS
![](https://img.shields.io/github/stars/OpenLMLab/MOSS.svg)
  * 简介：支持中英双语和多种插件的开源对话语言模型，MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。

* Luotuo-Chinese-LLM：
  * 地址：https://github.com/LC1332/Luotuo-Chinese-LLM
![](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM.svg)
  * 简介：囊括了一系列中文大语言模型开源项目，包含了一系列基于已有开源模型（ChatGLM, MOSS, LLaMA）进行二次微调的语言模型，指令微调数据集等。

* Linly：
  * 地址：https://github.com/CVI-SZU/Linly
![](https://img.shields.io/github/stars/CVI-SZU/Linly.svg)
  * 简介：提供中文对话模型 Linly-ChatFlow 、中文基础模型 Linly-Chinese-LLaMA 及其训练数据。 中文基础模型以 LLaMA 为底座，利用中文和中英平行增量预训练。项目汇总了目前公开的多语言指令数据，对中文模型进行了大规模指令跟随训练，实现了 Linly-ChatFlow 对话模型。

* ChatYuan
  * 地址：https://github.com/clue-ai/ChatYuan
![](https://img.shields.io/github/stars/clue-ai/ChatYuan.svg)
  * 简介：元语智能发布的一系列支持中英双语的功能型对话语言大模型，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

* ChatRWKV：
  * 地址：https://github.com/BlinkDL/ChatRWKV
![](https://img.shields.io/github/stars/BlinkDL/ChatRWKV.svg)
  * 简介：开源了一系列基于RWKV架构的Chat模型（包括英文和中文），发布了包括Raven，Novel-ChnEng，Novel-Ch与Novel-ChnEng-ChnPro等模型，可以直接闲聊及进行诗歌，小说等创作，包括7B和14B等规模的模型。

* CPM-Bee
  * 地址：https://github.com/OpenBMB/CPM-Bee
![](https://img.shields.io/github/stars/OpenBMB/CPM-Bee.svg)
  * 简介：一个完全开源、允许商用的百亿参数中英文基座模型。它采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型。

* TigerBot
  * 地址：https://github.com/TigerResearch/TigerBot
![](https://img.shields.io/github/stars/TigerResearch/TigerBot.svg)
  * 简介：一个多语言多任务的大规模语言模型(LLM)，开源了包括模型：TigerBot-7B, TigerBot-7B-base，TigerBot-180B，基本训练和推理代码，100G预训练数据，涵盖金融、法律、百科的领域数据以及API等。

* 书生·浦语
  * 地址：https://github.com/InternLM/InternLM-techreport
![](https://img.shields.io/github/stars/InternLM/InternLM-techreport.svg)
  * 简介：商汤科技、上海AI实验室联合香港中文大学、复旦大学和上海交通大学发布千亿级参数大语言模型“书生·浦语”（InternLM）。据悉，“书生·浦语”具有1040亿参数，基于“包含1.6万亿token的多语种高质量数据集”训练而成。

* Aquila
  * 地址：https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila
![](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg)
  * 简介：由智源研究院发布，Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

* baichuan-7B
  * 地址：https://github.com/baichuan-inc/baichuan-7B
![](https://img.shields.io/github/stars/baichuan-inc/baichuan-7B.svg)
  * 简介：由百川智能开发的一个开源可商用的大规模预训练语言模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。

* Anima
  * 地址：https://github.com/lyogavin/Anima
![](https://img.shields.io/github/stars/lyogavin/Anima.svg)
  * 简介：由艾写科技开发的一个开源的基于QLoRA的33B中文大语言模型，该模型基于QLoRA的Guanaco 33B模型使用Chinese-Vicuna项目开放的训练数据集guanaco_belle_merge_v1.0进行finetune训练了10000个step，基于Elo rating tournament评估效果较好。

* KnowLM
  * 地址：https://github.com/zjunlp/KnowLM
![](https://img.shields.io/github/stars/zjunlp/KnowLM.svg)
  * 简介：KnowLM项目旨在发布开源大模型框架及相应模型权重以助力减轻知识谬误问题，包括大模型的知识难更新及存在潜在的错误和偏见等。该项目一期发布了基于Llama的抽取大模型智析，使用中英文语料对LLaMA（13B）进行进一步全量预训练，并基于知识图谱转换指令技术对知识抽取任务进行优化。

* BayLing
  * 地址：https://github.com/ictnlp/BayLing
![](https://img.shields.io/github/stars/ictnlp/BayLing.svg)
  * 简介：一个具有增强的跨语言对齐的通用大模型，由中国科学院计算技术研究所自然语言处理团队开发。百聆（BayLing）以LLaMA为基座模型，探索了以交互式翻译任务为核心进行指令微调的方法，旨在同时完成语言间对齐以及与人类意图对齐，将LLaMA的生成能力和指令跟随能力从英语迁移到其他语言（中文）。在多语言翻译、交互翻译、通用任务、标准化考试的测评中，百聆在中文/英语中均展现出更好的表现。百聆提供了在线的内测版demo，以供大家体验。

* YuLan-Chat
  * 地址：https://github.com/ictnlp/BayLing
![](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Chat.svg)
  * 简介：YuLan-Chat是中国人民大学GSAI研究人员开发的基于聊天的大语言模型。它是在LLaMA的基础上微调开发的，具有高质量的英文和中文指令。 YuLan-Chat可以与用户聊天，很好地遵循英文或中文指令，并且可以在量化后部署在GPU（A800-80G或RTX3090）上。

###  2. <a name='Application'></a>Application

#### 2.1 垂直领域微调

##### 医疗

* DoctorGLM：
  * 地址：https://github.com/xionghonglin/DoctorGLM
![](https://img.shields.io/github/stars/xionghonglin/DoctorGLM.svg)
  * 简介：基于 ChatGLM-6B的中文问诊模型，通过中文医疗对话数据集进行微调，实现了包括lora、p-tuningv2等微调及部署

* BenTsao：
  * 地址：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese
![](https://img.shields.io/github/stars/SCIR-HI/Huatuo-Llama-Med-Chinese.svg)
  * 简介：开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对LLaMA进行了指令微调，提高了LLaMA在医疗领域的问答效果。

* BianQue：
  * 地址：https://github.com/scutcyr/BianQue
![](https://img.shields.io/github/stars/scutcyr/BianQue.svg)
  * 简介：一个经过指令与多轮问询对话联合微调的医疗对话大模型，基于ClueAI/ChatYuan-large-v2作为底座，使用中文医疗问答指令与多轮问询对话混合数据集进行微调。
  
* HuatuoGPT：
  * 地址：https://github.com/FreedomIntelligence/HuatuoGPT
![](https://img.shields.io/github/stars/FreedomIntelligence/HuatuoGPT.svg)
  * 简介：开源了经过中文医学指令精调/指令微调(Instruct-tuning)的一个GPT-like模型

* Med-ChatGLM：
  * 地址：https://github.com/SCIR-HI/Med-ChatGLM
![](https://img.shields.io/github/stars/SCIR-HI/Med-ChatGLM.svg)
  * 简介：基于中文医学知识的ChatGLM模型微调，微调数据与BenTsao相同。

* QiZhenGPT：
  * 地址：https://github.com/CMKRG/QiZhenGPT
![](https://img.shields.io/github/stars/CMKRG/QiZhenGPT.svg)
  * 简介：该项目利用启真医学知识库构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调，大幅提高了模型在中文医疗场景下效果，首先针对药品知识问答发布了评测数据集，后续计划优化疾病、手术、检验等方面的问答效果，并针对医患问答、病历自动生成等应用展开拓展。

* ChatMed：
  * 地址：https://github.com/michael-wzhu/ChatMed
![](https://img.shields.io/github/stars/michael-wzhu/ChatMed.svg)
  * 简介：该项目推出ChatMed系列中文医疗大规模语言模型，模型主干为LlaMA-7b并采用LoRA微调，具体包括ChatMed-Consult : 基于中文医疗在线问诊数据集ChatMed_Consult_Dataset的50w+在线问诊+ChatGPT回复作为训练集；ChatMed-TCM : 基于中医药指令数据集ChatMed_TCM_Dataset，以开源的中医药知识图谱为基础，采用以实体为中心的自指令方法(entity-centric self-instruct)，调用ChatGPT得到2.6w+的围绕中医药的指令数据训练得到。

* XrayGLM，首个会看胸部X光片的中文多模态医学大模型：
  * 地址：https://github.com/WangRongsheng/XrayGLM
![](https://img.shields.io/github/stars/WangRongsheng/XrayGLM.svg)
  * 简介：该项目为促进中文领域医学多模态大模型的研究发展，发布了XrayGLM数据集及模型，其在医学影像诊断和多轮交互对话上显示出了非凡的潜力。

* MeChat，中文心理健康支持对话大模型：
  * 地址：https://github.com/qiuhuachuan/smile
![](https://img.shields.io/github/stars/qiuhuachuan/smile.svg)
  * 简介：该项目开源的中文心理健康支持通用模型由 ChatGLM-6B LoRA 16-bit 指令微调得到。数据集通过调用gpt-3.5-turbo API扩展真实的心理互助 QA为多轮的心理健康支持多轮对话，提高了通用语言大模型在心理健康支持领域的表现，更加符合在长程多轮对话的应用场景。
 
* MedicalGPT
  * 地址：https://github.com/shibing624/MedicalGPT
![](https://img.shields.io/github/stars/shibing624/MedicalGPT.svg)
  * 简介：训练医疗大模型，实现包括二次预训练、有监督微调、奖励建模、强化学习训练。发布中文医疗LoRA模型shibing624/ziya-llama-13b-medical-lora，基于Ziya-LLaMA-13B-v1模型，SFT微调了一版医疗模型，医疗问答效果有提升，发布微调后的LoRA权重。

##### 法律

* LaWGPT：基于中文法律知识的大语言模型
  * 地址：https://github.com/pengxiao-song/LaWGPT
![](https://img.shields.io/github/stars/pengxiao-song/LaWGPT.svg)
  * 简介：该系列模型在通用中文基座模型（如 Chinese-LLaMA、ChatGLM 等）的基础上扩充法律领域专有词表、大规模中文法律语料预训练，增强了大模型在法律领域的基础语义理解能力。在此基础上，构造法律领域对话问答数据集、中国司法考试数据集进行指令精调，提升了模型对法律内容的理解和执行能力。

* LexiLaw：中文法律大模型
  * 地址：https://github.com/CSHaitao/LexiLaw
![](https://img.shields.io/github/stars/CSHaitao/LexiLaw.svg)
  * 简介：LexiLaw 是一个基于 ChatGLM-6B微调的中文法律大模型，通过在法律领域的数据集上进行微调。该模型旨在为法律从业者、学生和普通用户提供准确、可靠的法律咨询服务，包括具体法律问题的咨询，还是对法律条款、案例解析、法规解读等方面的查询。

* Lawyer LLaMA：中文法律LLaMA
  * 地址：https://github.com/AndrewZhe/lawyer-llama
![](https://img.shields.io/github/stars/AndrewZhe/lawyer-llama.svg)
  * 简介：开源了一系列法律领域的指令微调数据和基于LLaMA训练的中文法律大模型的参数。Lawyer LLaMA 首先在大规模法律语料上进行了continual pretraining。在此基础上，借助ChatGPT收集了一批对中国国家统一法律职业资格考试客观题（以下简称法考）的分析和对法律咨询的回答，利用收集到的数据对模型进行指令微调，让模型习得将法律知识应用到具体场景中的能力。

##### 金融

* Cornucopia（聚宝盆）：基于中文金融知识的LLaMA微调模型
  * 地址：https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese
![](https://img.shields.io/github/stars/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese.svg)
  * 简介：开源了经过中文金融知识指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过中文金融公开数据+爬取的金融数据构建指令数据集，并在此基础上对LLaMA进行了指令微调，提高了 LLaMA 在金融领域的问答效果。基于相同的数据，后期还会利用GPT3.5 API构建高质量的数据集，另在中文知识图谱-金融上进一步扩充高质量的指令数据集。

* BBT-FinCUGE-Applications
  * 地址：https://github.com/ssymmetry/BBT-FinCUGE-Applications
![](https://img.shields.io/github/stars/ssymmetry/BBT-FinCUGE-Applications.svg)
  * 简介：开源了中文金融领域开源语料库BBT-FinCorpus，中文金融领域知识增强型预训练语言模型BBT-FinT5及中文金融领域自然语言处理评测基准CFLEB。

* XuanYuan（轩辕）：首个千亿级中文金融对话模型
  * 地址：https://github.com/Duxiaoman-DI/XuanYuan
![](https://img.shields.io/github/stars/Duxiaoman-DI/XuanYuan.svg)
  * 简介：轩辕是国内首个开源的千亿级中文对话大模型，同时也是首个针对中文金融领域优化的千亿级开源对话大模型。轩辕在BLOOM-176B的基础上针对中文通用领域和金融领域进行了针对性的预训练与微调，它不仅可以应对通用领域的问题，也可以解答与金融相关的各类问题，为用户提供准确、全面的金融信息和建议。

##### 教育

* 桃李（Taoli）：
  * 地址：https://github.com/blcuicall/taoli
![](https://img.shields.io/github/stars/blcuicall/taoli.svg)
  * 简介：一个在国际中文教育领域数据上进行了额外训练的模型。项目基于目前国际中文教育领域流通的500余册国际中文教育教材与教辅书、汉语水平考试试题以及汉语学习者词典等，构建了国际中文教育资源库，构造了共计 88000 条的高质量国际中文教育问答数据集，并利用收集到的数据对模型进行指令微调，让模型习得将知识应用到具体场景中的能力。

##### 数学

* chatglm-maths：
  * 地址：https://github.com/yongzhuo/chatglm-maths
![](https://img.shields.io/github/stars/yongzhuo/chatglm-maths.svg)
  * 简介：基于chatglm-6b微调/LORA/PPO/推理的数学题解题大模型, 样本为自动生成的整数/小数加减乘除运算, 可gpu/cpu部署，开源了训练数据集等。

##### 文化

* Firefly：
  * 地址：https://github.com/yangjianxin1/Firefly
![](https://img.shields.io/github/stars/yangjianxin1/Firefly.svg)
  * 简介：中文对话式大语言模型，构造了许多与中华文化相关的数据，以提升模型这方面的表现，如对联、作诗、文言文翻译、散文、金庸小说等。

#### 2.2 LangChain应用

* Chinese-LangChain：
  * 地址：https://github.com/yanqiangmiffy/Chinese-LangChain
![](https://img.shields.io/github/stars/yanqiangmiffy/Chinese-LangChain.svg)
  * 简介：基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成（包括互联网检索结果接入）

* langchain-ChatGLM：
  * 地址：https://github.com/imClumsyPanda/langchain-ChatGLM
![](https://img.shields.io/github/stars/imClumsyPanda/langchain-ChatGLM.svg)
  * 简介：基于本地知识库的 ChatGLM 等大语言模型应用实现

#### 2.3 外部挂件应用

* wenda：
  * 地址：https://github.com/wenda-LLM/wenda
![](https://img.shields.io/github/stars/wenda-LLM/wenda.svg)
  * 简介：一个LLM调用平台。为小模型外挂知识库查找和设计自动执行动作，实现不亚于于大模型的生成能力。

* JittorLLMs：
  * 地址：https://github.com/Jittor/JittorLLMs
![](https://img.shields.io/github/stars/Jittor/JittorLLMs.svg)
  * 简介：计图大模型推理库：笔记本没有显卡也能跑大模型，具有成本低，支持广，可移植，速度快等优势。

* fastllm：
  * 地址：https://github.com/ztxz16/fastllm
![](https://img.shields.io/github/stars/ztxz16/fastllm.svg)
  * 简介：纯c++的全平台llm加速库，chatglm-6B级模型单卡可达10000+token / s，支持moss, chatglm, baichuan模型，手机端流畅运行。

* WebCPM
  * 地址：https://github.com/thunlp/WebCPM
![](https://img.shields.io/github/stars/thunlp/WebCPM.svg)
  * 简介：一个支持可交互网页搜索的中文大模型。 

* GPT Academic：
  * 地址：https://github.com/binary-husky/gpt_academic
![](https://img.shields.io/github/stars/binary-husky/gpt_academic.svg)
  * 简介：为GPT/GLM提供图形交互界面，特别优化论文阅读润色体验，支持并行问询多种LLM模型，支持清华chatglm等本地模型。兼容复旦MOSS, llama, rwkv, 盘古等。

* ChatALL：
  * 地址：https://github.com/sunner/ChatALL
![](https://img.shields.io/github/stars/sunner/ChatALL.svg)
  * 简介：ChatALL（中文名：齐叨）可以把一条指令同时发给多个 AI，可以帮助用户发现最好的回答。

* CreativeChatGLM：
  * 地址：https://github.com/ypwhs/CreativeChatGLM
![](https://img.shields.io/github/stars/ypwhs/CreativeChatGLM.svg)
  * 简介：可以使用修订和续写的功能来生成创意内容，可以使用“续写”按钮帮 ChatGLM 想一个开头，并让它继续生成更多的内容，你可以使用“修订”按钮修改最后一句 ChatGLM 的回复。

###  3. <a name='Dataset'></a>Dataset

* RefGPT：基于RefGPT生成大量真实和定制的对话数据集
  * 地址：https://github.com/ziliwangnlp/RefGPT
![](https://img.shields.io/github/stars/ziliwangnlp/RefGPT.svg)
  * 数据集说明：包括RefGPT-Fact和RefGPT-Code两部分，其中RefGPT-Fact给出了5万中文的关于事实性知识的多轮对话，RefGPT-Code给出了3.9万中文编程相关的多轮对话数据。

* COIG
  * 地址：https://huggingface.co/datasets/BAAI/COIG
  * 数据集说明：维护了一套无害、有用且多样化的中文指令语料库，包括一个人工验证翻译的通用指令语料库、一个人工标注的考试指令语料库、一个人类价值对齐指令语料库、一个多轮反事实修正聊天语料库和一个 leetcode 指令语料库。

* generated_chat_0.4M：
  * 地址：https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M
  * 数据集说明：包含约40万条由BELLE项目生成的个性化角色对话数据，包含角色介绍。但此数据集是由ChatGPT产生的，未经过严格校验，题目或解题过程可能包含错误。

* alpaca_chinese_dataset：
  * 地址：https://github.com/hikariming/alpaca_chinese_dataset
![](https://img.shields.io/github/stars/hikariming/alpaca_chinese_dataset.svg)
  * 数据集说明：根据斯坦福开源的alpaca数据集进行中文翻译，并再制造一些对话数据

* Alpaca-CoT：
  * 地址：https://github.com/PhoebusSi/Alpaca-CoT
![](https://img.shields.io/github/stars/PhoebusSi/Alpaca-CoT.svg)
  * 数据集说明：统一了丰富的IFT数据（如CoT数据，目前仍不断扩充）、多种训练效率方法（如lora，p-tuning）以及多种LLMs，三个层面上的接口，打造方便研究人员上手的LLM-IFT研究平台。

* pCLUE：
  * 地址：https://github.com/CLUEbenchmark/pCLUE
![](https://img.shields.io/github/stars/CLUEbenchmark/pCLUE.svg)
  * 数据集说明：基于提示的大规模预训练数据集，用于多任务学习和零样本学习。包括120万训练数据，73个Prompt，9个任务。

* firefly-train-1.1M：
  * 地址：https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
  * 数据集说明：23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万

* BELLE-data-1.5M：
  * 地址：https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M
![](https://img.shields.io/github/stars/LianjiaTech/BELLE/tree/main/data/1.5M.svg)
  * 数据集说明：通过self-instruct生成，使用了中文种子任务，以及openai的text-davinci-003接口,涉及175个种子任务

* Chinese Scientific Literature Dataset：
  * 地址：https://github.com/ydli-ai/csl
![](https://img.shields.io/github/stars/ydli-ai/csl.svg)
  * 数据集说明：中文科学文献数据集（CSL），包含 396,209 篇中文核心期刊论文元信息 （标题、摘要、关键词、学科、门类）以及简单的prompt

* Chinese medical dialogue data：
  * 地址：https://github.com/Toyhom/Chinese-medical-dialogue-data
![](https://img.shields.io/github/stars/Toyhom/Chinese-medical-dialogue-data.svg)
  * 数据集说明：中文医疗对话数据集，包括：<Andriatria_男科> 94596个问答对 <IM_内科> 220606个问答对 <OAGD_妇产科> 183751个问答对 <Oncology_肿瘤科> 75553个问答对 <Pediatric_儿科> 101602个问答对 <Surgical_外科> 115991个问答对 总计 792099个问答对。

* Huatuo-26M：
  * 地址：https://github.com/FreedomIntelligence/Huatuo-26M
![](https://img.shields.io/github/stars/FreedomIntelligence/Huatuo-26M.svg)
  * 数据集说明：Huatuo-26M 是一个中文医疗问答数据集，此数据集包含了超过2600万个高质量的医疗问答对，涵盖了各种疾病、症状、治疗方式、药品信息等多个方面。Huatuo-26M 是研究人员、开发者和企业为了提高医疗领域的人工智能应用，如聊天机器人、智能诊断系统等需要的重要资源。

* Alpaca-GPT-4:
  * 地址：https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
![](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM.svg)
  * 数据集说明：Alpaca-GPT-4 是一个使用 self-instruct 技术，基于 175 条中文种子任务和 GPT-4 接口生成的 50K 的指令微调数据集。

* InstructionWild
  * 地址：https://github.com/XueFuzhao/InstructionWild
![](https://img.shields.io/github/stars/XueFuzhao/InstructionWild.svg)
  * 数据集说明：InstructionWild 是一个从网络上收集自然指令并过滤之后使用自然指令结合 ChatGPT 接口生成指令微调数据集的项目。主要的指令来源：Twitter、CookUp.AI、Github 和 Discard。

* ShareChat
  * 地址：https://paratranz.cn/projects/6725
  * 数据集说明：一个倡议大家一起翻译高质量 ShareGPT 数据的项目。
  * 项目介绍：清洗/构造/翻译中文的ChatGPT数据，推进国内AI的发展，人人可炼优质中文 Chat 模型。本数据集为ChatGPT约九万个对话数据，由ShareGPT API获得（英文68000，中文11000条，其他各国语言）。项目所有数据最终将以 CC0 协议并入 Multilingual Share GPT 语料库。

* Guanaco
  * 地址：https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
  * 数据集说明：一个使用 Self-Instruct 的主要包含中日英德的多语言指令微调数据集。

* chatgpt-corpus
  * 地址：https://github.com/PlexPt/chatgpt-corpus
![](https://img.shields.io/github/stars/PlexPt/chatgpt-corpus.svg)
  * 数据集说明：开源了由 ChatGPT3.5 生成的300万自问自答数据，包括多个领域，可用于用于训练大模型。

* SmileConv
  * 地址：https://github.com/qiuhuachuan/smile
![](https://img.shields.io/github/stars/qiuhuachuan/smile.svg)
  * 数据集说明：数据集通过ChatGPT改写真实的心理互助 QA为多轮的心理健康支持多轮对话（single-turn to multi-turn inclusive language expansion via ChatGPT），该数据集含有56k个多轮对话，其对话主题、词汇和篇章语义更加丰富多样，更加符合在长程多轮对话的应用场景。

###  4. <a name='Evaluation'></a>Evaluation

* FlagEval （天秤）大模型评测体系及开放平台
  * 地址：https://github.com/FlagOpen/FlagEval
![](https://img.shields.io/github/stars/FlagOpen/FlagEval.svg)
  * 简介：旨在建立科学、公正、开放的评测基准、方法、工具集，协助研究人员全方位评估基础模型及训练算法的性能，同时探索利用AI方法实现对主观评测的辅助，大幅提升评测的效率和客观性。FlagEval （天秤）创新构建了“能力-任务-指标”三维评测框架，细粒度刻画基础模型的认知能力边界，可视化呈现评测结果。

* C-Eval: 构造中文大模型的知识评估基准：
  * 地址：https://github.com/SJTU-LIT/ceval
![](https://img.shields.io/github/stars/SJTU-LIT/ceval.svg)
  * 简介：构造了一个覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集。此外还给出了当前主流中文LLM的评测结果。

* SuperCLUElyb: SuperCLUE琅琊榜
  * 地址：https://github.com/CLUEbenchmark/SuperCLUElyb
![](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUElyb.svg)
  * 简介：中文通用大模型匿名对战评价基准，这是一个中文通用大模型对战评价基准，它以众包的方式提供匿名、随机的对战。他们发布了初步的结果和基于Elo评级系统的排行榜。

* GAOKAO-Bench: 
  * 地址：https://github.com/OpenLMLab/GAOKAO-Bench
![](https://img.shields.io/github/stars/OpenLMLab/GAOKAO-Bench.svg)
  * 简介：GAOKAO-bench是一个以中国高考题目为数据集，测评大模型语言理解能力、逻辑推理能力的测评框架，收集了2010-2022年全国高考卷的题目，其中包括1781道客观题和1030道主观题，构建起GAOKAO-bench的数据部分。

* AGIEval: 
  * 地址：https://github.com/microsoft/AGIEval
![](https://img.shields.io/github/stars/microsoft/AGIEval.svg)
  * 简介：由微软发布的一项新型基准测试，这项基准选取20种面向普通人类考生的官方、公开、高标准往常和资格考试，包括普通大学入学考试（中国高考和美国 SAT 考试）、法学入学考试、数学竞赛、律师资格考试、国家公务员考试等等。

* Xiezhi: 
  * 地址：https://github.com/mikegu721/xiezhibenchmark
![](https://img.shields.io/github/stars/mikegu721/xiezhibenchmark.svg)
  * 简介：由复旦大学发布的一个综合的、多学科的、能够自动更新的领域知识评估Benchmark，包含了哲学、经济学、法学、教育学、文学、历史学、自然科学、工学、农学、医学、军事学、管理学、艺术学这13个学科门类，24万道学科题目，516个具体学科，249587道题目。

* Open LLM Leaderboard：
  * 地址：https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
  * 简介：由HuggingFace组织的一个LLM评测榜单，目前已评估了较多主流的开源LLM模型。评估主要包括AI2 Reasoning Challenge, HellaSwag, MMLU, TruthfulQA四个数据集上的表现，主要以英文为主。

* chinese-llm-benchmark：
  * 地址：https://github.com/jeinlee1991/chinese-llm-benchmark
![](https://img.shields.io/github/stars/jeinlee1991/chinese-llm-benchmark.svg)
  * 简介：中文大模型能力评测榜单：覆盖百度文心一言、chatgpt、阿里通义千问、讯飞星火、belle / chatglm6b 等开源大模型，多维度能力评测。不仅提供能力评分排行榜，也提供所有模型的原始输出结果！

* Safety-Prompts：
  * 地址：https://github.com/thu-coai/Safety-Prompts
![](https://img.shields.io/github/stars/thu-coai/Safety-Prompts.svg)
  * 简介：由清华大学提出的一个关于LLM安全评测benchmark，包括安全评测平台等，用于评测和提升大模型的安全性，囊括了多种典型的安全场景和指令攻击的prompt。

* MMCU：
  * 地址：https://github.com/Felixgithub2017/MMCU
![](https://img.shields.io/github/stars/Felixgithub2017/MMCU.svg)
  * 简介：该项目提供对中文大模型语义理解能力的测试，评测方式、评测数据集、评测记录都公开，确保可以复现。该项目旨在帮助各位研究者们评测自己的模型性能，并验证训练策略是否有效。

* PromptCBLUE: 中文医疗场景的LLM评测基准
  * 地址：https://github.com/michael-wzhu/PromptCBLUE
![](https://img.shields.io/github/stars/michael-wzhu/PromptCBLUE.svg)
  * 简介：为推动LLM在医疗领域的发展和落地，由华东师范大学联合阿里巴巴天池平台，复旦大学附属华山医院，东北大学，哈尔滨工业大学（深圳），鹏城实验室与同济大学推出PromptCBLUE评测基准, 将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务,形成首个中文医疗场景的LLM评测基准。


###  5. <a name='Tutorial'></a>Tutorial

* 面向开发者的 LLM 入门课程：
  * 地址：https://github.com/datawhalechina/prompt-engineering-for-developers
![](https://img.shields.io/github/stars/datawhalechina/prompt-engineering-for-developers.svg)
  * 简介：一个中文版的大模型入门教程，围绕吴恩达老师的大模型系列课程展开，主要包括：吴恩达《ChatGPT Prompt Engineering for Developers》课程中文版，吴恩达《Building Systems with the ChatGPT API》课程中文版，吴恩达《LangChain for LLM Application Development》课程中文版等。

* 提示工程指南:
  * 地址：https://www.promptingguide.ai/zh
  * 简介：该项目基于对大语言模型的浓厚兴趣，编写了这份全新的提示工程指南，介绍了大语言模型相关的论文研究、学习指南、模型、讲座、参考资料、大语言模型能力以及与其他与提示工程相关的工具。

* LangChain 🦜️🔗 中文网，跟着LangChain一起学LLM/GPT开发：
  * 地址：https://www.langchain.asia
  * 简介：Langchain的中文文档，由是两个在LLM创业者维护，希望帮助到从刚进入AI应用开发的朋友们。

* LLMs九层妖塔：
  * 地址：https://github.com/km1994/LLMsNineStoryDemonTower
![](https://img.shields.io/github/stars/km1994/LLMsNineStoryDemonTower.svg)
  * 简介：ChatGLM、Chinese-LLaMA-Alpaca、MiniGPT-4、FastChat、LLaMA、gpt4all等实战与经验。

* HuggingLLM：
  * 地址：https://github.com/datawhalechina/hugging-llm
![](https://img.shields.io/github/stars/datawhalechina/hugging-llm.svg)
  * 简介：介绍 ChatGPT 原理、使用和应用，降低使用门槛，让更多感兴趣的非NLP或算法专业人士能够无障碍使用LLM创造价值。

* OpenAI Cookbook：
  * 地址：https://github.com/openai/openai-cookbook
![](https://img.shields.io/github/stars/openai/openai-cookbook.svg)
  * 简介：该项目是OpenAI提供的使用OpenAI API的示例和指导，其中包括如何构建一个问答机器人等教程，能够为从业人员开发类似应用时带来指导。

* awesome-chatgpt-prompts-zh：
  * 地址：https://github.com/PlexPt/awesome-chatgpt-prompts-zh
![](https://img.shields.io/github/stars/PlexPt/awesome-chatgpt-prompts-zh.svg)
  * 简介：该项目是ChatGPT 中文调教指南。各种场景使用指南。学习怎么让它听你的话，对指令构造可以提供一些参考。

* llm-action：
  * 地址：https://github.com/liguodongiot/llm-action
![](https://img.shields.io/github/stars/liguodongiot/llm-action.svg)
  * 简介：该项目提供了一系列LLM实战的教程和代码，包括LLM的训练、推理、微调以及LLM生态相关的一些技术文章等。

* LLMsPracticalGuide：
  * 地址：https://github.com/Mooler0410/LLMsPracticalGuide
![](https://img.shields.io/github/stars/Mooler0410/LLMsPracticalGuide.svg)
  * 简介：该项目提供了关于LLM的一系列实用指南与资源精选列表（包括LLM发展历程、示例、论文等）。
 
 ###  6. <a name='Related Repository'></a>Related Repository
 
* FindTheChatGPTer：
  * 地址：https://github.com/chenking2020/FindTheChatGPTer
![](https://img.shields.io/github/stars/chenking2020/FindTheChatGPTer.svg)
  * 简介：ChatGPT爆火，开启了通往AGI的关键一步，本项目旨在汇总那些ChatGPT的开源平替们，包括文本大模型、多模态大模型等，为大家提供一些便利。

* LLM_reviewer：
  * 地址：https://github.com/SpartanBin/LLM_reviewer
![](https://img.shields.io/github/stars/SpartanBin/LLM_reviewer.svg)
  * 简介：总结归纳近期井喷式发展的大语言模型，以开源、规模较小、可私有化部署、训练成本较低的‘小羊驼类’模型为主。

* Awesome-AITools：
  * 地址：https://github.com/ikaijua/Awesome-AITools
![](https://img.shields.io/github/stars/ikaijua/Awesome-AITools.svg)
  * 简介：收藏整理了AI相关的实用工具、评测和相关文章。

* open source ChatGPT and beyond：
  * 地址：https://github.com/SunLemuria/open_source_chatgpt_list
![](https://img.shields.io/github/stars/SunLemuria/open_source_chatgpt_list.svg)
  * 简介：This repo aims at recording open source ChatGPT, and providing an overview of how to get involved, including: base models, technologies, data, domain models, training pipelines, speed up techniques, multi-language, multi-modal, and more to go.

* Awesome Totally Open Chatgpt：
  * 地址：https://github.com/nichtdax/awesome-totally-open-chatgpt
![](https://img.shields.io/github/stars/nichtdax/awesome-totally-open-chatgpt.svg)
  * 简介：This repo record a list of totally open alternatives to ChatGPT.

* Awesome-LLM：
  * 地址：https://github.com/Hannibal046/Awesome-LLM
![](https://img.shields.io/github/stars/Hannibal046/Awesome-LLM.svg)
  * 简介：This repo is a curated list of papers about large language models, especially relating to ChatGPT. It also contains frameworks for LLM training, tools to deploy LLM, courses and tutorials about LLM and all publicly available LLM checkpoints and APIs.

* DecryptPrompt：
  * 地址：https://github.com/DSXiangLi/DecryptPrompt
![](https://img.shields.io/github/stars/DSXiangLi/DecryptPrompt.svg)
  * 简介：总结了Prompt&LLM论文，开源数据&模型，AIGC应用。

* Awesome Pretrained Chinese NLP Models：
  * 地址：https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models
![](https://img.shields.io/github/stars/lonePatient/awesome-pretrained-chinese-nlp-models.svg)
  * 简介：收集了目前网上公开的一些高质量中文预训练模型。

* ChatPiXiu：
  * 地址：https://github.com/catqaq/ChatPiXiu
![](https://img.shields.io/github/stars/catqaq/ChatPiXiu.svg)
  * 简介：该项目旨在打造全面且实用的ChatGPT模型库和文档库。当前V1版本梳理了包括：相关资料调研+通用最小实现+领域/任务适配等。

* LLM-Zoo：
  * 地址：https://github.com/DAMO-NLP-SG/LLM-Zoo
![](https://img.shields.io/github/stars/DAMO-NLP-SG/LLM-Zoo.svg)
  * 简介：该项目收集了包括开源和闭源的LLM模型，具体包括了发布时间，模型大小，支持的语种，领域，训练数据及相应论文/仓库等。

* LLMs-In-China：
  * 地址：https://github.com/wgwang/LLMs-In-China
![](https://img.shields.io/github/stars/wgwang/LLMs-In-China.svg)
  * 简介：该项目旨在记录中国大模型发展情况，同时持续深度分析开源开放的大模型以及数据集的情况。

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

