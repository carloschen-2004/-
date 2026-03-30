"""
知识库
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()     # 处理字符串前后的空格和回车
            if line == md5_str:
                return True         # 已处理过
        return False


def save_md5(md5_str: str):
    """将传入的md5字符串 记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    """根据md5字符串 返回对应的文件名"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()     # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容（传入即将要转换的字节）
    md5_hex = md5_obj.hexdigest()       # 得到md5的十六进制字符串
    return md5_hex 



class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)   # 创建数据库本地存储文件夹
        self.chroma = Chroma(
            collection_name=config.collection_name,     # 数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=config.api_key),
            persist_directory=config.persist_directory,     # 数据库本地存储文件夹
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,) 
    
    def upload_by_str(self, data: str, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "【跳过】内容已经在知识库中"
        if len(data) > config.max_split_char_number:
            data_chunks = self.splitter.split_text(data)
        else:   
            data_chunks = [data]
        
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "陈实",
        }

        self.chroma.add_texts(
            data_chunks, 
            metadatas = [metadata for _ in data_chunks]
        )

        save_md5(md5_hex)
        return "【成功】内容已存入知识库"
   
   
if __name__ == "__main__":
    service = KnowledgeBaseService()
    r = service.upload_by_str("hello world", "test.txt")
    print(r)
