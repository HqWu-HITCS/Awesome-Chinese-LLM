### 金融类大模型的应用

* BBT-Fin  (2023-02-18)
  * Paper: https://arxiv.org/abs/2302.09432
  * Code: https://github.com/ssymmetry/BBT-FinCUGE-Applications
  * 基础模型: FinT5 (0.2B, 1B)
  * 数据:
    * BBT-FinCorpus (Base: 4GB, Large: 16GB)
      * 公司公告、研究报告
        * 东方财富
      * 财经新闻
        * 新浪金融
        * 腾讯金融
        * 凤凰金融
        * 36氪
        * 虎嗅
      * 社交媒体
        * 东方财富 - 股吧
        * 雪球
    * 评测
      * BBT-CFLEB
        * FinNA - 金融新闻摘要数据集 (24k, 3k, 3k)
        * FinQA - 金融新闻公告事件问答数据集 (16k, 2k, 2k)
        * FinNL - 金融新闻分类数据集 (8k, 1k, 1k)
        * FinRE - 金融新闻关系抽取数据集 (7.5k, 1.5k, 3.7k)
        * FinFE - 金融社交媒体文本情绪分类数据集 (8k,1k,1k)
        * FinNSP - 金融负面消息及主体判定数据集 (4.8k, 0.6k, 0.6k)
  * 机构:
    * 复旦大学

* 聚宝盆 (Cornucopia)  (2023-05-07)
  * Code: https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese
  * License: Apache-2.0
  * 基座模型:
    * LLaMA-7B, Chinese-LLaMA-7B
  * 数据:
    * 14M 指令数据
    * 中文知识图谱-金融
    * CFLEB金融数据集
  * 算力:
    * NVIDIA A100 SXM 80GB
    * 10 epochs, batch size 96
  * 机构: 中科院成都计算机应用研究所

* 轩辕 (XuanYuan 2.0)  (2023-05-19)
  * Paper:
    * https://arxiv.org/abs/2305.12002
    * https://arxiv.org/abs/2305.11952
    * https://arxiv.org/abs/2305.14471
  * Code: https://github.com/Duxiaoman-DI/XuanYuan
  * 模型:
    * https://huggingface.co/xyz-nlp/XuanYuan2.0
  * 基座模型:
    * BLOOM-176B
      * 70 layers
      * 14336 hidden size
      * 112 attention heads
    * BLOOM-7B
      * 30 layers
      * 4096 hidden size
      * 32 attention heads
  * 数据:
    * 13B tokens
      * 预训练数据来自互联网
      * 指令训练数据
        * 对Self-Instruct
        * 对金融领域的数据的Self-QA得到信息
  * 算力:
    * 8 x NVIDIA A100 80GB + DeepSpeed
    * ZeRO (stage-1)
    * batch-size: 2048
  * 机构: 度小满 (原百度金融)


* 貔貅(PIXIU/FinMA)  (2023-06-08)
  * Paper: https://arxiv.org/abs/2306.05443
  * Code: https://github.com/chancefocus/PIXIU
  * License: MIT
  * 模型: LLaMA-7B, LLaMA-30B
  * 数据:
    * 预训练
    * 指令微调
      * 136K 指令数据
        * 5 类任务、9个数据集
          * 金融情感分析
            * Financial Phrase Bank
            * FiQA-SA
          * 新闻标题分类
            * Gold 新闻标题数据集
          * 命名实体识别
            * FIN
          * 问答
            * FinQA
            * ConvFinQA
          * 股价变动预测
            * BigData22
            * ACL18
            * CIKM18
    * 测试
      * FLARE
  * 算力:
    * max length: 2048
    * FinMA-7B
      * 8 x NVIDIA A100 40GB
      * 15 epochs, batch size 32
    * FinMA-30B
      * 128 x NVIDIA A100 40GB
      * 20 epochs, batch size 24
  * 院校:
    * 武汉大学
    * 中山大学
    * 西南交通大学
    * University of Florida

* FinGPT  (2023-06-09)
  * Paper: https://arxiv.org/abs/2306.06031
  * Code: https://github.com/AI4Finance-Foundation/FinGPT
  * License: MIT
  * 基座模型:
    * FinGPT v1
      * ChatGLM-6B + LoRA
    * FinGPT v2
      * LLaMA-7B + LoRA
  * 数据:
    * 金融新闻
      * Reuters
      * CNBC
      * Yahoo Finance
      * 东方财富
      * Financial Modeling Prep
    * 社交媒体
      * Twitter
      * Facebook
      * Reddit
      * 新浪微博
    * 财报
      * SEC
      * 证券交易所官网
    * 趋势
      * Seeking Alpha
      * Google Trends
    * 学术数据集
  * 算力:
    * $300/训练
  * 机构:
    * Columbia University
    * New York University (Shanghai)

