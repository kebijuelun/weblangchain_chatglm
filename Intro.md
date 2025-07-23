# 仓库简介

本文档帮助新加入的开发者快速了解项目结构以及推荐的学习路径。

## 仓库整体结构
```
.
├── main.py              # FastAPI 后端入口，定义 RAG 逻辑
├── env.sh               # 环境变量示例
├── nextjs/              # 前端（Next.js + React）代码
├── Dockerfile / fly.toml / Procfile / vercel.json  # 部署配置
├── assets/              # 示例图片/GIF
├── test_chatglm.py      # ChatGLM 调用示例
├── test_tavily.py       # Tavily 检索示例
└── README.md / README_CN.md  # 中英文说明
```

### 后端核心
- `main.py` 构建多种可配置的检索器（Tavily、Google、You、Kay.ai 等），并创建完整的检索与生成链。
- 默认将本地部署的 ChatGLM3 作为 OpenAI API 兼容服务使用，地址由 `openai_api_base` 指定。
- 通过 `langserve` 将链挂载到 `/chat` 路由供前端调用。

### 前端核心
- 位于 `nextjs/` 目录，主要组件在 `nextjs/app/components/`。
- `ChatWindow.tsx` 负责对话逻辑和 SSE 流式处理，支持从下拉框选择不同的模型或检索器。

### 运行与配置
- `README.md` 与 `README_CN.md` 详细说明本地安装和运行步骤。
- 各种 API Key 等配置位于 `env.sh`，修改后即可启动。

## 学习建议
1. **熟悉 LangChain**：阅读官方文档中关于 Retriever、Contextual Compression 与 Runnable 链式调用的部分，以理解 `main.py` 中的链构建方式。
2. **了解 Next.js 与 React**：前端基于函数式组件和 Hooks，可参考 [Next.js 文档](https://nextjs.org/)。
3. **学习 LangServe**：项目通过 `langserve` 将链暴露为 API，了解其路由添加和流式接口的用法有助于自定义后台服务。
4. **研究部署流程**：若计划上线，可查看 `Dockerfile`、`fly.toml` 和 `Procfile` 中的配置，理解不同云平台的部署差异。

通过先理解后端的检索-压缩-生成流程，再结合前端 `ChatWindow` 组件查看如何发起请求与展示结果，可以更快熟悉整个系统的工作方式。
