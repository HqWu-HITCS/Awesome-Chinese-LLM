### 医学类大模型的应用

* ChatDoctor  (2023-03-24)
  * Paper: https://arxiv.org/abs/2303.14070
  * Code: https://github.com/Kent0n-Li/ChatDoctor
  * License: Apache 2.0
  * 基座模型：LLaMA-7B
  * 数据：
    * HealthCareMagic-100k
    * icliniq-10k
    * GenMedGPT-5k
    * disease database
  * 算力：
    * 6 x NVIDIA A100 GPUs, 18h, batch size 192, 3 epochs
  * 院校:
    * Department of Radiation Oncology, University of Texas Southwestern Medical Center, Dallas, USA
    * Department of Computer Science, University of Illinois at Urbana-Champaign, Illinois, USA
    * Department of Computer Science and Engineering, The Ohio State University, Columbus, USA
    * 杭州电子科技大学计算机学院

* DoctorGLM  (2023-04-03)
  * Paper: https://arxiv.org/abs/2304.01097
  * Code: https://github.com/xionghonglin/DoctorGLM
  * 基座模型: THUDM/chatglm-6b
  * 数据：
    * CMD
      * Surgical (116K)
      * Obstetrics and Gynecology (229K)
      * Pediatrics (117K)
      * Internal Medicine (307K)
      * Andriatria (113K)
    * MedDialog (3.4M)
    * ChatDoctor (5.4K)
    * HealthCareMagic (200K)
  * 算力：
    * 1 x NVIDIA A100 GPU 80GB, 13h
  * 院校：
    * 上海科技大学
    * 上海交通大学
    * 复旦大学华山医院

* MedicalGPT-zh  (2023-04-08)
  * Code: https://github.com/MediaBrain-SJTU/MedicalGPT-zh
  * License: Apache 2.0
  * 基座模型: ChatGLM-6B
  * 数据：
    * 28科室的中文医疗共识与临床指南文本
      * 情景对话 (52K)
      * 知识问答 (130K)
  * 算力：
    * 4 x NVIDIA 3090 GPUs
  * 院校：上海交通大学未来媒体网络协同创新中心

* Chinese-Vicuna-Medical  (2023-04-11)
  * Code: https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-medical.md
  * License: Apache 2.0
  * 基座模型：Chinese-Vicuna-7B
  * 数据：
    * cMedQA2
  * 算力：
    * 70w of data, 3 epochs, a 2080Ti about 200h

* 本草(BenTsao)  (2023-04-14)
  * 原名：华佗(HuaTuo)
  * Paper: https://arxiv.org/abs/2304.06975
  * Code: https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese
  * License: Apache 2.0
  * 基座模型：LLaMA-7B, Chinese-LLaMA-Alpaca, ChatGLM-6B
  * 数据：
    * 公开和自建的中文医学知识库，主要参考了cMeKG
    * 2023年关于肝癌疾病的中文医学文献，利用GPT3.5接口围绕医学文献多轮问答数据
  * 算力：
    * A100-SXM-80GB，10 epochs, 2h17m, batch_size=128
  * 院校：哈尔滨工业大学社会计算与信息检索研究中心健康智能组

* OpenBioMed  (2023-04-17)
  * Paper: https://arxiv.org/abs/2305.01523  (2023-04-17)
  * Paper: https://arxiv.org/abs/2306.04371  (2023-06-07)
  * Code: https://github.com/BioFM/OpenBioMed
  * License: MIT
  * 模型：BioMedGPT-1.6B
  * 数据：DeepDTA
  * 院校：清华大学计算机系

* ChatMed  (2023-04-19)
  * Code: https://github.com/michael-wzhu/ChatMed
  * License: Apache 2.0
  * 基座模型：LLaMA-7B + Chinese-LLaMA-Alpaca
  * 数据：
    * 中文医疗在线问诊数据集ChatMed_Consult_Dataset的50w+在线问诊+ChatGPT回复作为训练集
    * 中医药指令数据集ChatMed_TCM_Dataset
    * 中医药知识图谱
      * ChatGPT得到11w+的围绕中医药的指令数据
  * 算力：
    * 4 x NVIDIA 3090 GPUS
  * 院校：华东师范大学

* 扁鹊(BianQue)  (2023-04-22)
  * Code: https://github.com/scutcyr/BianQue
  * 基座模型：
    * v1: 元语智能
    * v2: ChatGLM-6B
  * 数据：
    * 中文医疗问答指令与多轮问询对话混合数据集包含了超过900万条样本
    * 扁鹊健康大数据BianQueCorpus
      * 扩充了药品说明书指令
      * 医学百科知识指令
      * ChatGPT蒸馏指令等数据
      * MedDialog-CN
      * IMCS-V2
      * CHIP-MDCFNPC
      * MedDG
      * cMedQA2
      * Chinese-medical-dialogue-data
  * 算力：
    * 8张 NVIDIA RTX 4090显卡上微调了1个epoch，大约16天
  * 院校：华南理工大学未来技术学院

* PMC-LLaMA  (2023-04-27)
  * Paper: https://arxiv.org/abs/2304.14454
  * Code: https://github.com/chaoyi-wu/PMC-LLaMA
  * 基座模型: LLaMA-13B
  * 数据：
    * PubmedCentral papers (4.8M)
  * 院校：上海交通大学未来媒体网络协同创新中心

* MeChat  (2023-04-30)
  * Paper: https://arxiv.org/abs/2305.00450
  * Code: https://github.com/qiuhuachuan/smile
  * License: CC0-1.0
  * Model: https://huggingface.co/qiuhuachuan/MeChat
  * 基座模型: ChatGLM-6B
  * 微调方式: LoRA
  * 数据:
    * PsyQA
  * 院校:
    * 西湖大学
    * 浙江大学

* 启真医学大模型  (2023-05-23)
  * Code: https://github.com/CMKRG/QiZhenGPT
  * License: GPL-3.0
  * 基座模型：ChatGLM-6B, CaMA-13B, Chinese-LLaMA-Plus-7B
  * 数据：
    * 启真医学知识库
      * 真实医患知识问答数据
      * 在启真医学知识库的药品文本知识基础上，通过对半结构化数据设置特定的问题模板构造的指令数据
    * 药品适应症评测数据集
  * 算力：
    * 7 x NVDIA A800 GPU 80GB
      * ChatGLM-6B, 16h20m (2500),
      * CaMA-13B, 54h30m (6000) / 114h46m (12400)
      * Chinese-LLaMA-Plus-7B, 40h56m (6000)
  * 院校：浙江大学

* XrayGLM  (2023-05-23)
  * Code: https://github.com/WangRongsheng/XrayGLM
  * License: CC BY-NC-SA 4.0
  * 基座模型: VisualGLM-6B
  * 数据:
    * MIMIC-CXR (377K Image + 227K Report)
    * Openl (6459 Image + 3955 Report)
  * 算力: 4 x NVIDIA A100 GPUs 80GB
  * 院校: 澳门理工大学应用科学学院

* 华佗GPT (HuaTuoGPT)  (2023-05-24)
  * Papser: https://arxiv.org/abs/2305.15075
  * Code: https://github.com/FreedomIntelligence/HuatuoGPT
  * License: Apache 2.0
  * 基座模型: BLOOMZ-7b1
  * 数据:
    * 用 ChatGPT 构成的指令数据集 (61K)
    * 真实的医生指令集 (70K)
    * ChatGPT 角色扮演医患对话数据集 (68K)
    * 真实的医患对话数据集 (26K)
  * 算力:
    * 8 x NVIDIA A100 GPUs, 3 epochs, 16000 steps,
  * 院校: 香港中文大学(深圳)

* MedicalGPT  (2023-06-05)
  * Code: https://github.com/shibing624/MedicalGPT
  * License: Apache 2.0
  * 基座模型：Ziya-LLaMA-13B-v1 等
  * 数据：
    * 医疗数据：
      * 240万条中文医疗数据集(包括预训练、指令微调和奖励数据集)：shibing624/medical
      * 22万条中文医疗对话数据集(华佗项目)：FreedomIntelligence/HuatuoGPT-sft-data-v1
    * 通用数据：
      * 50万条中文ChatGPT指令Belle数据集：BelleGroup/train_0.5M_CN
      * 100万条中文ChatGPT指令Belle数据集：BelleGroup/train_1M_CN
      * 5万条英文ChatGPT指令Alpaca数据集：50k English Stanford Alpaca dataset
      * 2万条中文ChatGPT指令Alpaca数据集：shibing624/alpaca-zh
      * 69万条中文指令Guanaco数据集(Belle50万条+Guanaco19万条)：Chinese-Vicuna/guanaco_belle_merge_v1.0
      * 5万条英文ChatGPT多轮对话数据集：RyokoAI/ShareGPT52K
      * 80万条中文ChatGPT多轮对话数据集：BelleGroup/multiturn_chat_0.8M
      * 116万条中文ChatGPT多轮对话数据集：fnlp/moss-002-sft-data
    * Reward Model datasets
      * 原版的oasst1数据集：OpenAssistant/oasst1
      * 2万条多语言oasst1的reward数据集：tasksource/oasst1_pairwise_rlhf_reward
      * 11万条英文hh-rlhf的reward数据集：Dahoas/full-hh-rlhf
      * 9万条英文reward数据集(来自Anthropic's Helpful Harmless dataset)：Dahoas/static-hh
      * 7万条英文reward数据集（来源同上）：Dahoas/rm-static
      * 7万条繁体中文的reward数据集（翻译自rm-static）liswei/rm-static-m2m100-zh
      * 7万条英文Reward数据集：yitingxie/rlhf-reward-datasets
      * 3千条中文知乎问答偏好数据集：liyucheng/zhihu_rlhf_3k
  * 作者：徐明

* ClinicalGPT  (2023-06-16)
  * Paper: https://arxiv.org/abs/2306.09968
  * 基座模型: BLOOM-7B
  * 数据:
    * cMedQA2 (120K: 10K RM + 4K RL)
    * cMedQA-KG (100K)
    * MD-EHR (100K)
    * MEDQA-MCMLE (34K)
    * MedDialog (100K)
  * 院校: 北京邮电大学

* 孙思邈(Sunsimiao)  (2023-06-21)
  * Code: https://github.com/thomas-yanxin/Sunsimiao
  * License: Apache 2.0
  * 基座模型:
    * Sunsimiao: baichuan-7B
    * Sunsimiao-6B: ChatGLM2-6B
  * 数据:
    * 十万级高质量的中文医疗数据
  * 院校: 华东理工大学信息科学与工程学院

* 神农(ShenNong-TCM)  (2023-06-25)
  * Code: https://github.com/michael-wzhu/ShenNong-TCM-LLM
  * License: Apache 2.0
  * 基座模型: Chinese-Alpaca-Plus-7B
  * 数据:
    * 中医药指令数据集 ShenNong_TCM_Dataset
      * 以开源的中医药知识图谱为基础
      * 调用ChatGPT得到11w+的围绕中医药的指令数据
  * 院校：华东师范大学
