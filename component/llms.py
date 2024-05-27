from langchain.schema import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI,OpenAI
from langchain.prompts import  PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from   component.embedding import Zhipuembedding,OpenAIembedding,HFembedding,Jinaembedding
from component.data_chunker import ReadFile
from component.databases import Vectordatabase
import os
import json
from typing import Dict, List, Optional, Tuple, Union
import PyPDF2

#把api_key放在环境变量中,可以在系统环境变量中设置，也可以在代码中设置
# import os
# os.environ['OPENAI_API_KEY'] = ''

class Openai_model:
    def __init__(self,model_name:str='gpt-3.5-turbo-instruct',temperature:float=0.9) -> None:
        
        #初始化大模型
        self.model_name=model_name
        self.temperature=temperature
        self.model=OpenAI(model=model_name,temperature=temperature)

        #加载向量数据库，embedding模型
        self.db=Vectordatabase()
        self.db.load_vector()
        self.embedding_model=Zhipuembedding()
        
    #定义chat方法
    def chat(self,question:str):

        #这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {info}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""


        info=self.db.query(question,self.embedding_model,1)

        prompt=PromptTemplate(template=template,input_variables=["question","info"]).format(question=question,info=info)

        res=self.model.invoke(prompt)


        return  res

