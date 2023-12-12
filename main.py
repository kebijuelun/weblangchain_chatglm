"""Main entrypoint for the app."""
import asyncio
import os
from operator import itemgetter
from typing import List, Optional, Sequence, Tuple, Union
from uuid import UUID

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatVertexAI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.retrievers import (
    ContextualCompressionRetriever,
    TavilySearchAPIRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.kay import KayAiRetriever
from langchain.retrievers.you import YouRetriever
from langchain.schema import Document
from langchain.schema.document import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Backup
from langchain.utilities import GoogleSearchAPIWrapper
from langserve import add_routes
from pydantic import BaseModel, Field

EN_PROMPT = False
if EN_PROMPT:
    RESPONSE_TEMPLATE = """\
    You are an expert researcher and writer, tasked with answering any question.

    Generate a comprehensive and informative, yet concise answer of 250 words or less for the \
    given question based solely on the provided search results (URL and content). You must \
    only use information from the provided search results. Use an unbiased and \
    journalistic tone. Combine search results together into a coherent answer. Do not \
    repeat text. Cite search results using [${{number}}] notation. Only cite the most \
    relevant results that answer the question accurately. Place these citations at the end \
    of the sentence or paragraph that reference them - do not put them all at the end. If \
    different results refer to different entities within the same name, write separate \
    answers for each entity. If you want to cite multiple results for the same sentence, \
    format it as `[${{number1}}] [${{number2}}]`. However, you should NEVER do this with the \
    same number - if you want to cite `number1` multiple times for a sentence, only do \
    `[${{number1}}]` not `[${{number1}}] [${{number1}}]`

    You should use bullet points in your answer for readability. Put citations where they apply \
    rather than putting them all at the end.

    If there is nothing in the context relevant to the question at hand, just say "Hmm, \
    I'm not sure." Don't try to make up an answer.

    Anything between the following `context` html blocks is retrieved from a knowledge \
    bank, not part of the conversation with the user.

    <context>
        {context}
    <context/>

    REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
    not sure." Don't try to make up an answer. Anything between the preceding 'context' \
    html blocks is retrieved from a knowledge bank, not part of the conversation with the \
    user.\
    """

    REPHRASE_TEMPLATE = """\
    Given the following conversation and a follow up question, rephrase the follow up \
    question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:"""
else:
    RESPONSE_TEMPLATE = """\
    您是一位专业的研究员和作家，负责回答任何问题。

    基于提供的搜索结果（URL 和内容），为给定的问题生成一个全面而且信息丰富、但简洁的答案，长度不超过 250 字。您必须只使用来自提供的搜索结果的信息。使用公正和新闻性的语气。将搜索结果合并成一个连贯的答案。不要重复文本。一定要使用 [${{number}}] 标记引用搜索结果，其中 number 代表搜索到的文档的 id 号，用 <doc id=\'x\'> 表示。只引用最相关的结果，以准确回答问题。将这些引用放在提到它们的句子或段落的末尾 - 不要全部放在末尾。如果不同的结果涉及同名实体的不同部分，请为每个实体编写单独的答案。如果要在同一句子中引用多个结果，请将其格式化为 [${{number1}}] [${{number2}}]。然而，您绝对不应该对相同的数字进行这样的操作 - 如果要在一句话中多次引用 number1，只需使用 [${{number1}}]，而不是 [${{number1}}] [${{number1}}]。

    为了使您的答案更易读，您应该在答案中使用项目符号。在适用的地方放置引用，而不是全部放在末尾。

    如果上下文中没有与当前问题相关的信息，只需说“嗯，我不确定。”不要试图编造答案。

    位于以下context HTML 块之间的任何内容都是从知识库中检索到的，而不是与用户的对话的一部分。
    <context>
        {context}
    <context/>

    请记住：一定要在回答的时候带上检索的内容来源标号。如果上下文中没有与问题相关的信息，只需说“嗯，我不确定。”不要试图编造答案。位于上述 'context' HTML 块之前的任何内容都是从知识库中检索到的，而不是与用户的对话的一部分。再次记住一定要在回答的时候带上检索的内容来源标号，比如回答的某句话的信息来源于第 <doc id=\'x\'> 的搜索结果，就在该句话的末尾使用 [${{x}}] 来进行标记。
    """

    REPHRASE_TEMPLATE = """\
    考虑到以下对话和一个后续问题，请将后续问题重新表达为独立的问题。

    聊天记录：
    {chat_history}
    后续输入：{question}
    独立问题："""


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )


class GoogleCustomSearchRetriever(BaseRetriever):
    search: Optional[GoogleSearchAPIWrapper] = None
    num_search_results = 3

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        if os.environ.get("GOOGLE_API_KEY", None) == None:
            raise Exception("No Google API key provided")

        if self.search == None:
            self.search = GoogleSearchAPIWrapper()

        # Get search questions
        print("Generating questions for Google Search ...")

        # Get urls
        print("Searching for relevant urls...")
        urls_to_look = []
        search_results = self.search_tool(query, self.num_search_results)
        print("Searching for relevant urls...")
        print(f"Search results: {search_results}")
        for res in search_results:
            if res.get("link", None):
                urls_to_look.append(res["link"])

        print(search_results)
        loader = AsyncHtmlLoader(urls_to_look)
        html2text = Html2TextTransformer()
        print("Indexing new urls...")
        docs = loader.load()
        docs = list(html2text.transform_documents(docs))
        for i in range(len(docs)):
            if search_results[i].get("title", None):
                docs[i].metadata["title"] = search_results[i]["title"]
        return docs


def get_retriever():
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    relevance_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    base_tavily_retriever = TavilySearchAPIRetriever(
        k=3,
        include_raw_content=True,
        include_images=True,
    )
    tavily_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_tavily_retriever
    )
    base_google_retriever = GoogleCustomSearchRetriever()
    google_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_google_retriever
    )
    base_you_retriever = YouRetriever(
        ydc_api_key=os.environ.get("YDC_API_KEY", "not_provided")
    )
    you_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_you_retriever
    )
    base_kay_retriever = KayAiRetriever.create(
        dataset_id="company",
        data_types=["10-K", "10-Q"],
        num_contexts=6,
    )
    kay_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_kay_retriever
    )
    base_kay_press_release_retriever = KayAiRetriever.create(
        dataset_id="company",
        data_types=["PressRelease"],
        num_contexts=6,
    )
    kay_press_release_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_kay_press_release_retriever,
    )
    return tavily_retriever.configurable_alternatives(
        # This gives this field an id
        # When configuring the end runnable, we can then use this id to configure this field
        ConfigurableField(id="retriever"),
        default_key="tavily",
        google=google_retriever,
        you=you_retriever,
        kay=kay_retriever,
        kay_press_release=kay_press_release_retriever,
    ).with_config(run_name="FinalSourceRetriever")


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def serialize_history(request: ChatRequest):
    chat_history = request.get("chat_history", [])
    converted_chat_history = []
    for message in chat_history:
        if message[0] == "human":
            converted_chat_history.append(HumanMessage(content=message[1]))
        elif message[0] == "ai":
            converted_chat_history.append(AIMessage(content=message[1]))
    return converted_chat_history


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever) | RunnableLambda(
        format_docs
    ).with_config(run_name="FormatDocumentChunks")
    _context = RunnableMap(
        {
            "context": retriever_chain.with_config(run_name="RetrievalChain"),
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(itemgetter("chat_history")).with_config(
                run_name="Itemgetter:chat_history"
            ),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
    )


dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    dir_path + "/" + ".google_vertex_ai_credentials.json"
)

has_google_creds = os.path.isfile(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

openai_api_base = "http://127.0.0.1:8000/v1"

llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    # model="gpt-4",
    streaming=True,
    temperature=0.1,
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="openai",
    chatglm=ChatOpenAI(model="chatglm3-6b", openai_api_base=openai_api_base)
)

if has_google_creds:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        # model="gpt-4",
        streaming=True,
        temperature=0.1,
    ).configurable_alternatives(
        # This gives this field an id
        # When configuring the end runnable, we can then use this id to configure this field
        ConfigurableField(id="llm"),
        default_key="openai",
    )

retriever = get_retriever()

chain = create_chain(llm, retriever)

add_routes(
    app, chain, path="/chat", input_type=ChatRequest, config_keys=["configurable"]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
