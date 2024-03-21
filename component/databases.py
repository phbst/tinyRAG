from tqdm import tqdm
import numpy as np 
from component.embedding import HFembedding,OpenAIembedding,Zhipuembedding,Jinaembedding
import os
import json
from typing import List




class VectorDB:
    def __init__(self,docs:List=[]) -> None:
        self.docs = docs

    def get_vector(self,EmbeddingModel)->List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.docs):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors
    def persist(self,path:str='db')->None:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)


    def load_vector(self,path:str='db')->None:
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float],embedding_model) -> float:
        return embedding_model.compare_v(vector1, vector2)
    
    def query(self, query: str, EmbeddingModel, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector,EmbeddingModel)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()