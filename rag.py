from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableWithMessageHistory
from file_history_store import get_history
import config_data as config


def printPrompt(prompt):
    print("===========")
    print(prompt)
    print("===========")
    return prompt  

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name, dashscope_api_key=config.api_key)
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
            ("system", "并且我提供用户对话的历史记录,如下: "),
            MessagesPlaceholder("history"),
            ("user","请回答用户的问题：{input}")
        ])
        self.chat_model = ChatTongyi(model_name=config.chat_model_name, api_key=config.api_key)
        self.chain = self.__get_chain()
    
    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(documents):
            if not documents:
                return "无相关参考资料"
            formatted_str = ""
            for doc in documents:
                formatted_str += f"文档片段：{doc.page_content} \n 文档元数据: {doc.metadata} \n \n"
            return formatted_str

        # 从输入中提取查询文本用于检索
        def extract_query(inputs: dict) -> str:
            """提取用户查询文本"""
            if isinstance(inputs, dict):
                # 支持两种格式: {"input": str} 和 {"input": {"input": str, "history": []}}
                input_val = inputs.get("input", "")
                if isinstance(input_val, dict):
                    return input_val.get("input", "")
                return input_val
            return str(inputs)

        # 准备prompt的输入
        def prepare_prompt(inputs: dict) -> dict:
            """准备prompt模板所需的输入"""
            return {
                "context": inputs.get("context", ""),
                "history": inputs.get("history", []),
                "input": extract_query(inputs)
            }

        chain = (
            RunnableParallel(
                {
                    "context": RunnablePassthrough.assign(
                        query=lambda x: extract_query(x)
                    ).assign(
                        context=lambda x: retriever.invoke(x["query"])
                    ).assign(
                        context=lambda x: format_document(x["context"])
                    ) | (lambda x: x["context"]),
                    "history": lambda x: x.get("history", []),
                    "input": lambda x: extract_query(x),
                }
            )
            | prepare_prompt
            | self.prompt_template
            | printPrompt
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

if __name__ == "__main__":
    # session_id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input":"夏天适合穿什么衣服"}, session_config)
    print(res)