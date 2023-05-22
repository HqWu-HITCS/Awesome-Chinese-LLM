# Awesome-Chinese-LLM

自ChatGPT为代表的大语言模型（Large Language Model, LLM）出现以后，由于其惊人的类通用人工智能（AGI）的能力，掀起了新一轮自然语言处理领域的研究和应用的浪潮。尤其是以ChatGLM、LLaMA等平民玩家都能跑起来的较小规模的LLM开源之后，业界涌现了非常多基于LLM的二次微调或应用的案例。本项目旨在收集和梳理中文LLM相关的开源模型、应用和数据集等资料。

1. [Model](#Model)
2. [Application](#Application)
3. [Dataset](#Dataset)


##  1. <a name='Model'></a>Model

* ChatGLM：https://github.com/THUDM/ChatGLM-6B

* Moss：https://github.com/OpenLMLab/MOSS

* Chinese-LLaMA-Alpaca：https://github.com/ymcui/Chinese-LLaMA-Alpaca

* alpaca-lora：https://github.com/tloen/alpaca-lora

* Panda：https://github.com/dandelionsllm/pandallm

* Luotuo-Chinese-LLM：https://github.com/LC1332/Luotuo-Chinese-LLM


##  2. <a name='Application'></a>Application

* Chinese-LangChain：https://github.com/yanqiangmiffy/Chinese-LangChain

* langchain-ChatGLM：https://github.com/imClumsyPanda/langchain-ChatGLM

* BELLE：https://github.com/LianjiaTech/BELLE

* DoctorGLM：https://github.com/xionghonglin/DoctorGLM

* BenTsao：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese

* Med-ChatGLM：https://github.com/SCIR-HI/Med-ChatGLM

* GPT Academic：https://github.com/binary-husky/gpt_academic

* wenda：https://github.com/wenda-LLM/wenda

* LaWGPT：https://github.com/pengxiao-song/LaWGPT

* Firefly：https://github.com/yangjianxin1/Firefly


##  3. <a name='Dataset'></a>Dataset

* alpaca_chinese_dataset：
  * 地址：https://github.com/hikariming/alpaca_chinese_dataset
  * 数据集说明：alpaca数据集进行中文翻译，并再制造一些对话数据

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


