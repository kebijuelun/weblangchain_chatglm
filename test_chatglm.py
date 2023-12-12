from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# default endpoint_url for a local deployed ChatGLM api server
openai_api_base = "http://127.0.0.1:8000/v1"

llm = ChatOpenAI(model="chatglm3-6b", openai_api_base=openai_api_base)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "北京和上海两座城市有什么不同？"

print(llm_chain.run(question))
