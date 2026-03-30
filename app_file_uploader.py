"""
基于Streamlit完成WEB网页上传服务
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService
import time


# 添加网页标题
st.title("知识库更新服务")

uploaded_file = st.file_uploader(
    "请上传TXT文件",
    type=["txt"],
    accept_multiple_files=False, # 仅允许上传一个文件
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploaded_file is not None:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024 # KB
    
    st.subheader(f"文件名: {file_name}")
    st.write(f"格式: {file_type}, 大小: {file_size:.2f} KB")

    text = uploaded_file.getvalue().decode("utf-8") # 读取文件内容并解码为字符串
    st.write(f"文件内容预览:\n{text[:100]}...")

    with st.spinner("载入知识库中..."):
        time.sleep(1) 
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
