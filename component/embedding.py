
## 这个组件，能下载的embedding模型都使用离线的embedding，不能下载的就使用api

import numpy as np
from transformers import AutoModel
from numpy.linalg import norm
from langchain.embeddings.openai import OpenAIEmbeddings
from zhipuai import ZhipuAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from typing import List


#Embedding类
class HFembedding:

#初始化embedding，如果是离线模型就下载下来，传入模型路径
    def __init__(self, path:str=''):
        self.path = path
        self.embedding=HuggingFaceEmbeddings(model_name=path)

#对字符串进行编码，传入字符串，输出一个向量
    def get_embedding(self,content:str=''):
        return self.embedding.embed_query(content)

#对两个字符串求相似度，使用embedding模型进行编码，再使用编码后的向量进行余弦相似度求值 
    def compare(self, text1: str, text2: str):
        embed1=self.embedding.embed_query(text1) 
        embed2=self.embedding.embed_query(text2)
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

#对两个向量进行相似度求值，余弦相似度求值 
    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OpenAIembedding:
    
    def __init__(self, path:str=''):
        self.path = path
        self.embedding=OpenAIEmbeddings()
    
    def get_embedding(self,content:str=''):
        content = content.replace("\n", " ")
        return self.embedding.embed_query(content)
    
    def compare(self, text1: str, text2: str):
        embed1=self.embedding.embed_query(text1) 
        embed2=self.embedding.embed_query(text2)
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    
    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude



class Zhipuembedding:

    def __init__(self, path:str=''):
        	

        client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY")) 
        self.embedding_model=client


    def get_embedding(self,content:str=''):
        response =self.embedding_model.embeddings.create(
            model="embedding-2", #填写需要调用的模型名称
            input=content #填写需要计算的文本内容,
        )
        return response.data[0].embedding

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
    def compare(self, text1: str, text2: str):
        embed1=self.embedding_model.embeddings.create(
            model="embedding-2", #填写需要调用的模型名称
            input=text1 #填写需要计算的文本内容,
        ).data[0].embedding

        embed2=self.embedding_model.embeddings.create(
            model="embedding-2", #填写需要调用的模型名称
            input=text2 #填写需要计算的文本内容,
        ).data[0].embedding

        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    



class Jinaembedding:

    def __init__(self, path:str='jinaai/jina-embeddings-v2-base-zh'):
        self.path = path
        self.embedding_model=AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True) 
    
    def get_embedding(self,content:str=''):
        return self.embedding_model.encode([content])[0]
    
    def compare(self, text1: str, text2: str):
        
        cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
        embeddings = self.embedding_model.encode([text1, text2])
        return cos_sim(embeddings[0], embeddings[1])

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
