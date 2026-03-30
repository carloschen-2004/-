# 智能客服系统

参考“黑马程序员大模型RAG与Agent智能体项目实战教程”制作的项目：

基于检索增强生成（RAG）技术的智能客服问答系统，专注于服装尺码推荐、洗涤养护及颜色搭配咨询。

## 技术架构

### 核心栈

- **LLM**: 通义千问 qwen3-max (阿里云DashScope)
- **Embedding**: text-embedding-v4 (阿里云DashScope)
- **向量数据库**: ChromaDB (本地持久化存储)
- **应用框架**: Streamlit + LangChain
- **语言**: Python 3.10+

### 系统架构图

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  RAG Service │────▶│   Qwen3     │
│   Web UI    │     │  (rag.py)    │     │    LLM      │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │
                    │ Vector Store │
                    └──────────────┘
```

## 项目结构

```
RAG/
├── app.py                    # Streamlit聊天主界面
├── rag.py                    # RAG核心服务（检索+生成）
├── knowledge_base.py         # 知识库管理与向量化
├── vector_stores.py          # 向量存储接口封装
├── file_history_store.py     # 聊天历史持久化
├── app_file_uploader.py      # 文件上传模块
├── config_data.py            # 配置管理
├── requirements.txt          # 依赖清单
├── data/                     # 知识库文档目录
└── chroma_db/                # 向量数据库存储
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置API密钥

在 `config_data.py` 中设置阿里云DashScope API密钥：

```python
DASHSCOPE_API_KEY = "your-api-key-here"
```

### 启动服务

```bash
streamlit run app.py
```

### 页面示例
<img src="image.png" width="600" />          