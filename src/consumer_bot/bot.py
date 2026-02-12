# 模块职责：RAG 服务端
# - 构建检索上下文（优先 chromadb，回退离线 BM25，再次回退空上下文）
# - 统一 DeepSeek/OpenAI 的模型与 BASE_URL 解析，添加自动降级重试
# - 通过 LangServe 暴露链路到 /consumer_ai
# 
# 运行入口：pdm run bot
# 相关环境变量：
# - CHROMA_COLLECTION：chromadb 集合名（默认 consumer_docs）
# - HF_ENDPOINT：HuggingFace 镜像（可选）
# - OPENAI_API_KEY / DEEPSEEK_API_KEY：模型密钥
# - OPENAI_API_BASE / OPENAI_BASE_URL / DEEPSEEK_API_BASE / DEEPSEEK_BASE_URL：模型基础地址
# - DEEPSEEK_MODEL / DS_MODEL / OPENAI_MODEL：模型名（DeepSeek 场景统一为 deepseek-chat）
# 
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import json
import chromadb


def format_docs(docs):
    """将检索结果 Document 列表拼接为提示词上下文。
    - 输入：List[Document]
    - 输出：字符串（以两个换行分隔段落）
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    """构建 RAG LCEL 链。
    流程：
    1) 优先连接 chromadb（有数据则用向量检索）
    2) 回退到离线 BM25（存在 offline_docs.json）
    3) 最终回退为空上下文
    输出：{"context","sources","question"} → Prompt → LLM → Parser
    """
    # 1) 优先尝试连接 chromadb 持久化集合；失败则标记为 None
    coll = None
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            coll = client.get_collection(name=os.getenv("CHROMA_COLLECTION", "consumer_docs"))
        except Exception:
            coll = client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "consumer_docs"))
    except Exception:
        coll = None
    if coll and coll.count() > 0:
        ef_cache = {"ef": None}
        def get_ctx_src(question: str):
            """向量检索，返回上下文与引用来源。
            - k 由环境变量 RETRIEVAL_K 控制，默认 4
            - sources 取自 metadata.source/url，并去重
            """
            try:
                if ef_cache["ef"] is None:
                    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                    ef_cache["ef"] = SentenceTransformerEmbeddingFunction(model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
                coll_with_ef = client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "consumer_docs"), embedding_function=ef_cache["ef"])
                k = int(os.getenv("RETRIEVAL_K", "4"))
                r = coll_with_ef.query(query_texts=[question], n_results=k)
                docs_list = r.get("documents") or []
                metas_list = r.get("metadatas") or []
                texts = docs_list[0] if docs_list else []
                metas = metas_list[0] if metas_list else []
                context = "\n\n".join(texts) if texts else ""
                srcs = []
                for i, m in enumerate(metas[:k]):
                    src = (m or {}).get("source") or (m or {}).get("url") or ""
                    if src:
                        srcs.append(f"S{i+1}: {src}")
                sources = "; ".join(list(dict.fromkeys(srcs)))
                return {"context": context, "sources": sources}
            except Exception:
                return {"context": "", "sources": ""}
        context = RunnableLambda(lambda q: get_ctx_src(q)["context"])
        sources = RunnableLambda(lambda q: get_ctx_src(q)["sources"])
    elif os.path.exists("./data/offline_docs.json"):
        # 3) 回退离线 BM25：使用 offline_docs.json 构建检索器
        from langchain_core.documents import Document
        from langchain_community.retrievers import BM25Retriever
        with open("./data/offline_docs.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
            docs = [Document(page_content=r["page_content"], metadata=r.get("metadata") or {}) for r in raw]
        if docs:
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = int(os.getenv("RETRIEVAL_K", "4"))
            def get_ctx_src(question: str):
                """BM25 检索，返回上下文与引用来源。
                - k 同样由 RETRIEVAL_K 控制
                - sources 来自每条 Document 的 metadata
                """
                try:
                    result_docs = retriever.invoke(question)
                    context = format_docs(result_docs)
                    k = int(os.getenv("RETRIEVAL_K", "4"))
                    srcs = []
                    for i, d in enumerate(result_docs[:k]):
                        meta = d.metadata or {}
                        s = meta.get("source") or meta.get("url") or ""
                        if s:
                            srcs.append(f"S{i+1}: {s}")
                    sources = "; ".join(list(dict.fromkeys(srcs)))
                    return {"context": context, "sources": sources}
                except Exception:
                    return {"context": "", "sources": ""}
            context = RunnableLambda(lambda q: get_ctx_src(q)["context"])
            sources = RunnableLambda(lambda q: get_ctx_src(q)["sources"])
        else:
            retriever = RunnableLambda(lambda _: [])
            context = RunnableLambda(lambda _: "")
            sources = RunnableLambda(lambda _: "")
    else:
        # 4) 最终回退：无数据时返回空上下文
        context = RunnableLambda(lambda _: "")
        sources = RunnableLambda(lambda _: "")
    # 提示词模板：简短回答并允许“不知道”
    prompt_template_str = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to "
        "answer the question. If you don't know the answer, just say that you don't know. Use three sentences "
        "maximum and keep the answer concise.\n\nQuestion: {question}\n\nContext: {context}\n\nSources: {sources}\n\nAnswer:"
    )
    prompt = PromptTemplate.from_template(prompt_template_str)

    def resolve_chat_model():
        """统一模型名解析。
        - 若 base 指向 DeepSeek，则归一化为 deepseek-chat
        - 否则读取 OPENAI_MODEL，默认 gpt-4o-mini
        """
        base = (os.getenv("OPENAI_BASE_URL") or "").lower()
        if "deepseek" in base:
            m = os.getenv("DEEPSEEK_MODEL") or os.getenv("DS_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-chat"
            s = m.strip().lower().replace("_", "-")
            if s in ("deepseek", "deepseek-chat", "chat", "deepseekchat"):
                return "deepseek-chat"
            return m
        return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    def resolve_base_url():
        """统一 BASE_URL 读取，并在指向 DeepSeek 时自动补齐 /v1。"""
        base = (
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or ""
        )
        if "deepseek" in base and not base.rstrip("/").endswith("/v1"):
            return base.rstrip("/") + "/v1"
        return base

    def call_llm(input_messages):
        """LLM 调用包装。
        - 缺密钥直接返回 AIMessage 提示
        - DeepSeek 400 模型不存在时自动降级到 deepseek-reasoner
        - 错误信息包含 base_url 与 model 便于定位
        """
        from langchain_core.messages import AIMessage
        if not os.getenv("OPENAI_API_KEY"):
            return AIMessage(content="缺少 OPENAI_API_KEY，无法调用真实模型。")
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        bu = resolve_base_url()
        mm = resolve_chat_model()
        def invoke(m):
            return ChatOpenAI(model=m, temperature=0, api_key=key, base_url=bu).invoke(input_messages)
        try:
            return invoke(mm)
        except Exception as e:
            msg = str(e)
            if "Model Not Exist" in msg and "deepseek" in bu.lower():
                for alt in ("deepseek-chat", "deepseek-reasoner"):
                    if alt != mm:
                        try:
                            return invoke(alt)
                        except Exception:
                            pass
            return AIMessage(content=f"模型调用失败: {e}; base_url={bu}; model={mm}")

    # LLM 作为 LCEL 可运行单元
    llm = RunnableLambda(call_llm)

    # LCEL 链：拼接上下文→提示词→LLM→解析器
    return (
        {"context": context, "sources": sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# 加载 .env 与离线配置（HF）
load_dotenv()
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 构建链与应用
rag_chain = build_chain()
app = FastAPI(title="知识库智能助手", version="1.0")
add_routes(app, rag_chain, path="/consumer_ai")


if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", "8000")))
