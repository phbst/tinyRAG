from langchain.schema import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI,OpenAI
from langchain.prompts import  PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from   component.embedding import Zhipuembedding,OpenAIembedding,HFembedding,Jinaembedding
from component.data_chunker import ReadFile
from component.databases import VectorDB
import os
import json
from typing import Dict, List, Optional, Tuple, Union
import PyPDF2

#把api_key放在环境变量中,可以在系统环境变量中设置，也可以在代码中设置
# import os
# os.environ['OPENAI_API_KEY'] = ''

class Openai_model:
    def __init__(self,model_name:str='gpt-3.5-turbo-instruct',temperature:float=0.9) -> None:
        
        self.model_name=model_name
        self.temperature=temperature
        self.model=OpenAI(model=model_name,temperature=temperature)

        self.db=VectorDB()
        self.db.load_vector()
        self.embedding_model=Zhipuembedding()
        

    def chat(self,question:str):
        template="""question:{question}\n以下列表信息供你参考,如果你觉得它对回答问题没有帮助，你可以忽视它：\n info:{info}"""
        info=self.db.query(question,self.embedding_model,1)

        prompt=PromptTemplate(template=template,input_variables=["question","info"]).format(question=question,info=info)

        res=self.model.invoke(prompt)


        return  res

