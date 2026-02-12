# 知识库智能助手（consumer_bot）

一个基于 LangChain + chromadb + FastAPI/LangServe 的问答机器人，围绕“知识库”提供检索增强的问答能力。项目支持在线网页抓取与本地文档知识库两种数据来源，并适配 DeepSeek/OpenAI 的模型调用。

## 特性
- RAG 检索增强：优先使用 chromadb 向量检索，离线回退 BM25
- 数据构建灵活：支持在线页面抓取与本地目录 txt/md 加载
- 模型适配稳健：DeepSeek 模型名自动归一与降级重试，BASE_URL 自动补 /v1
- 服务对外统一：通过 LangServe 暴露链路，含 playground 与 /docs

## 技术栈
- Python ≥ 3.12，PDM 管理依赖与脚本
- LangChain（LCEL 链、加载器、检索器）
- chromadb（PersistentClient 持久化存储）
- FastAPI + LangServe（服务与交互）
- sentence-transformers（嵌入模型 all-MiniLM-L6-v2）

## 目录结构（关键文件）
- src/consumer_bot/data_prep.py：数据构建脚本（在线/本地加载→切割→持久化或离线回退）
- src/consumer_bot/bot.py：RAG 服务端（chromadb/BM25 检索→提示词→模型→解析）
- src/consumer_bot/quickstart.py：LangChain 最小对话链示例
- chroma_db/：chromadb 持久化目录（chroma.sqlite3 等）
- data/offline_docs.json：离线检索语料（网络不可用时生成）

## 安装与准备
1. 安装 PDM 并创建虚拟环境
2. 在项目根目录创建并填写 .env（示例见下）
3. 可选：设置 HuggingFace 镜像，便于嵌入模型下载

### .env 示例
```
OPENAI_API_KEY=你的密钥
# DeepSeek/OpenAI BASE_URL 兼容，任选其一即可（代码会自动补 /v1）
OPENAI_API_BASE=https://api.deepseek.com/v1
# 或 OPENAI_BASE_URL=https://api.deepseek.com/v1
# 或 DEEPSEEK_API_BASE=https://api.deepseek.com/v1
# 或 DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 模型名（DeepSeek 场景统一归一为 deepseek-chat）
DEEPSEEK_MODEL=deepseek-chat
# 可选：DS_MODEL/OPENAI_MODEL 也会被解析

# HuggingFace 镜像（可选，用于下载嵌入模型）
HF_ENDPOINT=https://hf-mirror.com/

# 其他可选
USER_AGENT=Mozilla/5.0 ...
CHROMA_COLLECTION=consumer_docs
HOST=localhost
PORT=8000
```

## 数据构建
项目支持两种数据来源：在线法规页面（默认）与本地目录。

### 在线页面（默认 URL 指向国务院公报条例）
- 干跑（仅打印数据，不写库）：
```
pdm run python -m consumer_bot.data_prep --dry-run
```
- 正式构建（写入 chromadb；网络不可用时生成 offline_docs.json）：
```
pdm run python -m consumer_bot.data_prep
```

### 本地目录（txt/md）
```
pdm run python -m consumer_bot.data_prep --docs-dir ./docs
```

### 构建后验证
```
pdm run python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_or_create_collection(name='consumer_docs'); print(col.count())"
```
返回 >0 表示 chromadb 已存储可检索的数据。

## 启动服务
```
pdm run bot
```
启动成功后：
- Playground：/consumer_ai/playground/
- 文档：/docs/
- 默认地址：http://localhost:8000/

如需指定端口：
```
$env:PORT='8001'; pdm run bot
```

## 快速入门示例
- 干跑（打印提示词消息，不调用模型）：
```
pdm run quickstart-dry -- --input "你好"
```
- 真实调用：
```
pdm run quickstart -- --input "你好"
```

## 检索与回退策略
- 优先 chromadb 向量检索（集合计数 > 0）
- 若向量库不可用但存在 offline_docs.json，则用 BM25 检索
- 若两者都不可用，返回空上下文，避免异常

## DeepSeek 接入说明
- 模型名自动归一：
  - deepseek、deepseek_chat、chat、deepseekchat → deepseek-chat
- BASE_URL 自动补齐：
  - 若包含 deepseek 且不以 /v1 结尾，自动补上 /v1
- 调用失败自动降级：
  - deepseek-chat → deepseek-reasoner
- 错误提示会包含 base_url 与 model，便于定位

## 常见问题
- “Model Not Exist”
  - 检查 BASE_URL 是否为 https://api.deepseek.com/v1
  - 检查模型名是否归一为 deepseek-chat（或允许 reasoner）
  - 确认密钥对应租户对该模型有权限
- 嵌入模型下载超时（HuggingFace）
  - 设置 HF_ENDPOINT=https://hf-mirror.com/
  - 或提前将 sentence-transformers/all-MiniLM-L6-v2 缓存到本地
- chromadb 集合计数为 0
  - 说明尚未写入向量：运行非干跑构建或使用本地目录模式
- BM25 ZeroDivisionError
  - 已在服务端对空语料做兜底，当前版本不会再抛错

## 参考法规
- 国务院令第778号《中华人民共和国消费者权益保护法实施条例》（2024年第10号国务院公报）
  - https://www.gov.cn/gongbao/2024/issue_11266/202404/content_6944108.html

## 免责声明
- 项目示例仅供学习与研发参考；法规解读以官方文本为准；请勿在生产环境中直接使用未经审核的数据与回答。
