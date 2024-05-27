import os
import PyPDF2
import tiktoken


# 用于数据切分时，判断字块的token长度，速度比较快
enc = tiktoken.get_encoding("cl100k_base")


class ReadFile:


    #传入文件夹路径
    def __init__(self, path):
        self.path = path
        
    #读取初始化类时传入的文件夹路径，返回该文件夹路径下所有文件的路径
    def readlist(self):
        file_list = []  
        for filepath, dirnames, filenames in os.walk(self.path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                    if filename.endswith(".md"):
                        file_list.append(os.path.join(filepath, filename))
                    elif filename.endswith(".txt"):
                        file_list.append(os.path.join(filepath, filename))
                    elif filename.endswith(".pdf"):
                        file_list.append(os.path.join(filepath, filename))    
        
        return file_list

    # 切分数据，传入一个字符串，返回一个字块列表
    @classmethod
    def chunk_content(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []
        curr_len = 0
        curr_chunk = ''
        lines = text.split('\n')
        for line in lines:
            line = line.replace(' ', '')
            line_len = len(enc.encode(line))
            if curr_len + line_len <= max_token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:]+line
                curr_len = line_len + cover_content
        if curr_chunk:
            chunk_text.append(curr_chunk)
        return chunk_text




    #读取文件内容，传入一个文件路径，返回该文件内容字符串
    @classmethod
    def read_file_content(cls, file_path: str):
        if file_path.endswith('.pdf'):
            return cls.read_pdf_content(file_path)
        elif file_path.endswith('.md'):
            return cls.read_md_content(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_txt_content(file_path)

    @classmethod
    def read_md_content(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @classmethod
    def read_pdf_content(cls, file_path: str):
        text=""
        with open(file_path, 'rb') as f:
            reader=PyPDF2.PdfReader(f)
            for num_page in range(len(reader.pages)):
                text+=reader.pages[num_page].extract_text()
        return text

    @classmethod
    def read_txt_content(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            

    #该类的整合函数，根据初始化类时的文件夹路径，读取所有内容进行切分，返回一个字块列表
    def get_all_chunk_content(self,max_len:int=600,cover_len:int=150):
        docs=[]
        for file in self.readlist():
            
            content=self.read_file_content(file)

            chunk_content=self.chunk_content(content,max_len,cover_len)

            docs.extend(chunk_content)

        return docs
    
         