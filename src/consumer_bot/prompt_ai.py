import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

def resolve_model():
    base = (
        (os.getenv("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "")
        .lower()
    )
    if "deepseek" in base:
        m = os.getenv("DEEPSEEK_MODEL") or os.getenv("DS_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-chat"
        s = m.strip().lower().replace("_", "-")
        if s in ("deepseek", "deepseek-chat", "chat", "deepseekchat"):
            return "deepseek-chat"
        return m
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

def resolve_base_url():
    base = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or ""
    if "deepseek" in base and not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    return base

load_dotenv()
key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=resolve_model(), temperature=0, api_key=key, base_url=resolve_base_url())
rag_chain = (llm | StrOutputParser())

app = FastAPI(title="prompt教学", version="1.0")
add_routes(app, rag_chain, path="/prompt_ai")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", "8000")))
