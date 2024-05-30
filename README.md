
<p align="center" style="font-size:20px">
    <br> 项目分析与代码解释 | <a href="README-optimazing.md" >项目后续优化方案</a>
</p>
<br>
<hr>

# 项目于优化阶段，目前只是一个最小的框架，后续持续更新迭代，直至部署于生产环境
# Hands on TinyRAG

## 什么是RAG？

LLM会产生误导性的 “幻觉”，依赖的信息可能过时，处理特定知识时效率不高，缺乏专业领域的深度洞察，同时在推理能力上也有所欠缺。

正是在这样的背景下，检索增强生成技术（Retrieval-Augmented Generation，RAG）应时而生，成为 AI 时代的一大趋势。

RAG 通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地缓解了幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，使得大型语言模型在实际应用中变得更加实用和可信。

此仓库用于学习大模型RAG的相关内容，目前为手搓实现，主要是llama-index和langchain不太好魔改。此仓库可以方便看论文的时候，实现一些小的实验。以下为本仓库的RAG整体框架图。


以下为笔者所构思的RAG实现过程，这里面主要包括包括三个基本步骤：

1. 索引 — 将文档库分割成较短的 Chunk，并通过编码器构建向量索引。

2. 检索 — 根据问题和 chunks 的相似度检索相关文档片段。

3. 生成 — 以检索到的上下文为条件，生成问题的回答。

# 项目结构
```
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
│  └─ 中华人民共和国消费者权益保护法.pdf
├─ db
│  ├─ doecment.json
│  └─ vectors.json
├─ image
│  └─ 5386440326a2c9c5a06b5758484d375.png
├─ push.bat
├─ README.md
├─ requirements.txt
└─ webdemo_by_gradio.ipynb

```
# QuickStrat

安装依赖，需要 Python 3.10 以上版本。

```bash
pip install -r requirements.txt
```

导入所使用的包

```python
from   component.embedding import Zhipuembedding,OpenAIembedding,HFembedding,Jinaembedding
from component.data_chunker import ReadFile
from component.databases import VectorDB

```

```python
import os
import json
from typing import Dict, List, Optional, Tuple, Union
import PyPDF2

```
建立数据库
```python
# 建立数据库
filter=ReadFile('./data')
docs=filter.get_all_chunk_content(200,150)
embedding_model=Zhipuembedding()
database=VectorDB(docs)
Vectors=database.get_vector(embedding_model)
database.persist()
```

如果有数据库那就按照如下代码：
```python
# 将向量和文档内容保存到db目录下，下次再用就可以直接加载本地的数据库
#加载向量数据库
text="项目结构"
embedding_model=Zhipuembedding()
db=VectorDB()
db.load_vector('./db')
result=db.query(text,embedding_model,10)
print(result)
```


# 实现细节
参考blog: https://zhuanlan.zhihu.com/p/688842148

# 最终启动demo结果如下:
<div align="center">
    <img src="./image/5386440326a2c9c5a06b5758484d375.png" alt="RAG" width="100%">
</div>



# 思考：
    中华人民共和国消费者权益保护法的目录回答其实是不全的，应该是切分数据的问题，
    可以把每一块的文本设置得更长，且相邻块之间的重叠覆盖范围更大
    
1. 避免关键信息不能完整被包含
2. 防止关键信息被切分开


# extra
 
中文文本嵌入使用ZhipuEmbedding,英文可以使用Openai,Huggingface

---


# 参考文献

| Name                                                         | Paper Link                                |
| ------------------------------------------------------------ | ----------------------------------------- |
| When Large Language Models Meet Vector Databases: A Survey   | [paper](http://arxiv.org/abs/2402.01763)  |
| Retrieval-Augmented Generation for Large Language Models: A Survey | [paper](https://arxiv.org/abs/2312.10997) |
| Learning to Filter Context for Retrieval-Augmented Generation | [paper](http://arxiv.org/abs/2311.08377)  |
| In-Context Retrieval-Augmented Language Models               | [paper](https://arxiv.org/abs/2302.00083) |


