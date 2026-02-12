# 模块职责：数据构建脚本
# - 从在线页面或本地目录加载文本
# - 切割文档片段并持久化到 chromadb（离线回退到 JSON）
#
# 使用示例：
# - 在线页面干跑：pdm run python -m consumer_bot.data_prep --dry-run
# - 在线页面构建：pdm run python -m consumer_bot.data_prep
# - 本地目录构建：pdm run python -m consumer_bot.data_prep --docs-dir ./docs
#
import os
import argparse
import bs4
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def run(url: str | None, persist_dir: str, chunk_size: int, chunk_overlap: int, dry_run: bool, docs_dir: str | None = None):
    """数据构建主流程。
    入参：
    - url：在线页面地址（与 docs_dir 互斥）
    - persist_dir：chromadb 持久化目录
    - chunk_size/chunk_overlap：文本切割参数
    - dry_run：干跑模式（不写库，仅打印规模）
    - docs_dir：本地目录（txt/md），与 url 互斥
    产出：
    - 成功写入 chromadb；失败时写入 data/offline_docs.json
    """
    # 1) 环境初始化：加载 .env 与 HF 离线镜像配置
    load_dotenv()
    ua = os.getenv("USER_AGENT") or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36"
    if not os.getenv("USER_AGENT"):
        os.environ["USER_AGENT"] = ua
    if os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    docs = []
    if docs_dir:
        # 2) 本地目录加载：txt/md 文件
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        dl = DirectoryLoader(docs_dir, glob="**/*.{txt,md}", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
        docs = dl.load()
    else:
        # 3) 在线页面加载：可选按 CSS 类筛选 <p>；为空时会回退
        from langchain_community.document_loaders import WebBaseLoader
        content_class = os.getenv("CONTENT_CLASS")
        if content_class and content_class.lower() not in ("all", "none"):
            loader = WebBaseLoader(
                web_path=url,
                bs_kwargs=dict(parse_only=bs4.SoupStrainer("p", class_=(content_class))),
                requests_kwargs={"headers": {"User-Agent": ua}},
            )
        else:
            loader = WebBaseLoader(
                web_path=url,
                requests_kwargs={"headers": {"User-Agent": ua}},
            )
        docs = loader.load()
    # 4) 文本切割：长文本分段以提升检索相关性
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    if dry_run:
        # 干跑：打印数据规模，不写库
        print(f"docs={len(docs)}")
        print(f"splits={len(splits)}")
        if splits:
            print(f"first_split_len={len(splits[0].page_content)}")
        return
    # 5) 在线页面为空的两级回退：整页→抓取所有 <p> 合并
    if not splits and not docs_dir:
        loader_all = WebBaseLoader(
            web_path=url,
            requests_kwargs={"headers": {"User-Agent": ua}},
        )
        docs_all = loader_all.load()
        splits = splitter.split_documents(docs_all)
        if not splits:
            resp = requests.get(url, headers={"User-Agent": ua}, timeout=30)
            resp.encoding = resp.apparent_encoding or resp.encoding
            soup = bs4.BeautifulSoup(resp.text, "html.parser")
            candidates = soup.find_all("p")
            text = "\n".join([p.get_text(strip=True) for p in candidates if p.get_text(strip=True)])
            if not text:
                print("no content extracted; abort")
                return
            from langchain_core.documents import Document
            docs_all = [Document(page_content=text, metadata={"source": url})]
            splits = splitter.split_documents(docs_all)
        if not splits:
            print("no content extracted; abort")
            return
    # 6) 写入 chromadb（失败则写入离线 JSON）
    try:
        ef = SentenceTransformerEmbeddingFunction(model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        client = chromadb.PersistentClient(path=persist_dir)
        coll = client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "consumer_docs"), embedding_function=ef)
        ids = [f"doc-{i}" for i in range(len(splits))]
        texts = [d.page_content for d in splits]
        metas = [d.metadata for d in splits]
        coll.add(documents=texts, ids=ids, metadatas=metas)
        print(f"persisted_to={persist_dir}, count={coll.count()}")
    except Exception:
        os.makedirs("./data", exist_ok=True)
        path = "./data/offline_docs.json"
        out = [{"page_content": d.page_content, "metadata": d.metadata} for d in splits]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"persisted_offline={path}, count={len(out)}")


def main():
    """命令行入口。
    - 默认 URL 指向国务院公报条例页
    - 支持本地目录模式与干跑模式
    """
    # 命令行参数：默认 URL 指向国务院公报条例页
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.gov.cn/gongbao/2024/issue_11266/202404/content_6944108.html")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--docs-dir", default=None)
    args = parser.parse_args()
    run(args.url if not args.docs_dir else None, args.persist_dir, args.chunk_size, args.chunk_overlap, args.dry_run, args.docs_dir)


if __name__ == "__main__":
    main()
