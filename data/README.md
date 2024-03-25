## 什么是RAG？

检索增强生成(RAG) 是**一种使用来自私有或专有数据源的信息来辅助文本生成的技术**。

它将检索模型（设计用于搜索大型数据集或知识库）和生成模型（例如大型语言模型(LLM)，此类模型会使用检索到的信息生成可供阅读的文本回复）结合在一起。

## 为什么需要RAG？

LLM会产生误导性的 “幻觉”，依赖的信息可能过时，处理特定知识时效率不高，缺乏专业领域的深度洞察，同时在推理能力上也有所欠缺。

正是在这样的背景下，检索增强生成技术（Retrieval-Augmented Generation，RAG）应时而生，成为 AI 时代的一大趋势。

## 项目地址

**GitHub地址：**

https://github.com/phbst/tinyRAG

[**https://github.com/phbst/tinyRAG**](https://github.com/phbst/tinyRAG)

**全手写的一个RAG应用。Langchain的大部分库会很方便，但是你不一定理解其中原理，所以代码尽可能展现基本算法，主打理解RAG的原理**

## 代码讲解（以下之构建了一个简单的RAG结构，深入可自行了解）

---

总览：

项目结构如下

component是RAG的组件，分为五大部分（数据切分，向量化，向量存储，大模型，链）

data用于存放需要嵌入的文件（兼容Pdf  TXT，md文件）

db用于存放向量化后的数据，也是数据库的加载路径

build.ipynb构建向量数据库

webdemo_by_gradio使用gradio基于嵌入的文件调用OpenAI的回答助手

```markdown

tinyRAG
├─ build.ipynb
├─ component
│  ├─ chain.py
│  ├─ databases.py
│  ├─ data_chunker.py
│  ├─ embedding.py
│  └─ llms.py
├─ data
│  ├─ dpcq.txt
│  ├─ README.md
│  ├─ 中华人民共和国消费者权益保护法.pdf
│  └─ 简历.pdf
├─ db
│  ├─ doecment.json
│  └─ vectors.json
├─ image
│  └─ 微信图片_20240322110029.png
├─ README.md
└─ webdemo_by_gradio.ipynb
```

**component**