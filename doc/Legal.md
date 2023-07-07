
### 法律类大模型的应用

* 獬豸(LawGPT_zh)  (2023-04-09)
  * Code: https://github.com/LiuHC0428/LAW-GPT
  * License: 
  * 基础模型: ChatGLM-6B
  * 数据:
    * 情景对话：真实的律师用户问答 (200K)
      * 利用ChatGPT根据CrimeKgAssitant的问答重新生成 (52K)
      * 根据中华人民共和国法律手册上最核心的9k法律条文，利用ChatGPT联想生成具体的情景问答 (92K)
    * 知识问答：法律知识问题的解释性回答
      * 法律领域的教科书，经典案例等数据
  * 算力: 4 x NVIDIA 3090
  * 院校: 上海交通大学

* LaWGPT  (2023-04-12)
  * Code: https://github.com/pengxiao-song/LaWGPT
  * License: GPL-3.0
  * 基础模型: Chinese-Alpaca-Plus-7B
  * 数据:
    * https://github.com/pengxiao-song/awesome-chinese-legal-resources
      * 官方数据
        * 中国检查网：起诉书等
        * 中国裁判文书网：裁决书、裁定书、决定书等
        * 司法部国家司法考试中心：行政法规库、法考真题等
        * 国家法律法规数据库：官方法律法规数据库
        * https://github.com/pengxiao-song/awesome-chinese-legal-resources/issues/2
      * 竞赛数据
        * 中国法律智能技术评测（CAIL）历年赛题数据
        * 中国法研杯司法人工智能挑战赛（LAIC）历年赛题数据
        * 百度知道法律问答数据集：约 3.6w 条法律问答数据，包括用户提问、网友回答、最佳回答
        * 法律知识问答数据集：约 2.3w 条法律问答数据
        * 中国司法考试试题数据集：约 2.6w 条中国司法考试数据集
      * 开源数据
        * LaWGPT 数据集 @pengxiao-song：包含法律领域专有词表、结构化罪名数据、高质量问答数据等
        * 法律罪名预测与机器问答 @liuhuanyong：包括罪名知识图谱、20w 法务问答数据等
        * 法律条文知识抽取 @liuhuanyong：包括法律裁判文书和犯罪案例
        * 中国法律手册 @LawRefBook：收集各类法律法规、部门规章案例等
      * 其他
        * 刑法最新罪名一览表：记录2021年最新刑法罪名

      * 中文裁判文书网公开法律文书数据
      * 司法考试数据
      * 中国检查网：起诉书等
  * 算力: 8 x NVIDIA Tesla V100 32GB, 24h/epoch (pre-training), 12/epoch (fine-tuning)
  * 院校: 南京大学

* LexiLaw  (2023-05-16)
  * Code: https://github.com/CSHaitao/LexiLaw
  * License: MIT
  * 基础模型: ChatGLM-6B
  * 数据:
    * 通用领域数据
      * 链家 BELLE-1.5M
    * 法律问答数据
      * LawGPT_zh :52k单轮问答数据和92k带有法律依据的情景问答
      * Lawyer LLaMA :法考数据和法律指令微调数据
      * 华律网问答数据 :20k高质量华律网问答数据
      * 法律知道 :百度知道收集的36k条法律问答数据
    * 法律法规和法律参考书籍
      * 法律法规: 包含刑法、民法、宪法、司法解释等法律法规
      * 法律参考书籍: JEC-QA数据集提供的法律参考书籍
    * 法律文书
      * 从法律裁判文书网收集50k法律文书
  * 算力: 7 x NVIDIA A100 GPUs 40GB
  * 院校: 清华大学

* Lawyer LLaMA  (2023-05-24)
  * Paper: https://arxiv.org/abs/2305.15062
  * Code: https://github.com/AndrewZhe/lawyer-llama
  * License: Apache-2.0
  * 基础模型: Chinese-Alpaca-Plus-13B
  * 数据:
    * JEC-QA中国法考数据集
      * 中国国家统一法律职业资格考试客观题
  * 院校: 北京大学

* 韩非(HanFei)  (2023-05-31)
  * Code: https://github.com/siat-nlp/HanFei
  * License: Apache-2.0
  * 基座模型: BLOOMZ-7B1
  * 数据:
    * 预训练
      * 案例、法规、起诉状、法律新闻 (60G, 2K token/条)
    * 微调
      * v1.0
        * 中文通用指令 (53k)
        * 中文法律指令 (41k)
        * 中文通用对话 (55k)
        * 中文法律对话 (56k)
        * 中文法律问答数据 (50k)
    * 评估
      * 法律问题
        * 包含劳动、婚姻等9个板块 (150)
  * 算力: 8 x NVIDIA A100/A800
  * 机构:
    * 中科院深圳先进院
    * 深圳市大数据研究院
    * 香港中文大学（深圳）

* ChatLaw  (2023-06-28)
  * Paper: https://arxiv.org/abs/2306.16092
  * Code: https://github.com/PKU-YuanGroup/ChatLaw
  * License: AGPL-3.0
  * 基础模型:
    * ChatLaw-13B: 姜子牙 Ziya-LLaMA-13B-v1
    * ChatLaw-33B: Anima-33B
  * 数据: 由论坛、新闻、法条、司法解释、法律咨询、法考题、判决文书组成，随后经过清洗、数据增强等来构造对话数据
  * 算力: multiple NVIDIA V100 GPUs
  * 院校: 北京大学
