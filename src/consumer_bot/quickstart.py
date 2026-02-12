# 模块职责：LangChain 快速入门示例
# - 构建最小对话链：提示词→模型→输出解析
# - 适配 DeepSeek 的 BASE_URL 与模型名，并在失败时尝试自动降级
#
# 使用示例：
# - 干跑查看提示词：pdm run quickstart-dry -- --input "你好"
# - 调用真实模型：pdm run quickstart -- --input "你好"
#
import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable


def build_prompt() -> ChatPromptTemplate:
    """构建最小提示词。
    - 包含 system 设定与 human 输入两条消息
    - human 注入变量 {input}
    """
    # 两条消息：system 设定 + human 输入
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )


def build_chain(prompt: ChatPromptTemplate) -> Runnable:
    """构建最小对话链。
    - 解析模型名与 BASE_URL（DeepSeek 自动补 /v1）
    - 失败时自动降级 deepseek-chat → deepseek-reasoner
    - 返回 Runnable（invoke(inputs) 获取字符串输出）
    """
    # 统一解析模型名与 BASE_URL，并补齐 DeepSeek /v1
    base = (os.getenv("OPENAI_BASE_URL") or "").lower()
    if "deepseek" in base:
        m = os.getenv("DEEPSEEK_MODEL") or os.getenv("DS_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-chat"
        s = m.strip().lower().replace("_", "-")
        if s in ("deepseek", "deepseek-chat", "chat", "deepseekchat"):
            m = "deepseek-chat"
    else:
        m = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    bu = (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("DEEPSEEK_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or ""
    )
    if "deepseek" in bu and not bu.rstrip("/").endswith("/v1"):
        bu = bu.rstrip("/") + "/v1"
    model = ChatOpenAI(
        model=m,
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=bu,
    )
    chain = prompt | model | StrOutputParser()
    def safe_invoke(inputs):
        # 自动降级重试：deepseek-chat → deepseek-reasoner
        try:
            return chain.invoke(inputs)
        except Exception as e:
            msg = str(e)
            if "Model Not Exist" in msg and "deepseek" in bu.lower():
                for alt in ("deepseek-chat", "deepseek-reasoner"):
                    if alt != m:
                        alt_model = ChatOpenAI(model=alt, temperature=0, api_key=os.getenv("OPENAI_API_KEY"), base_url=bu)
                        try:
                            return (prompt | alt_model | StrOutputParser()).invoke(inputs)
                        except Exception:
                            pass
            return f"模型调用失败: {e}"
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(lambda x: safe_invoke(x))


def main():
    """命令行入口。
    - 支持 --dry-run 打印提示词消息
    - 真实调用需设置 OPENAI_API_KEY/DEEPSEEK_API_KEY
    """
    # 加载 .env 并解析命令参数
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt = build_prompt()

    if args.dry_run:
        # 干跑：打印提示词消息，不调用模型
        pv = prompt.invoke({"input": args.input})
        for m in pv.to_messages():
            print(m.content)
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    # 执行链并打印结果
    chain = build_chain(prompt)
    result = chain.invoke({"input": args.input})
    print(result)


if __name__ == "__main__":
    main()
