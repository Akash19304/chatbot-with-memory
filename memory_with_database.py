from dotenv import load_dotenv
import os
load_dotenv()
import warnings
warnings.filterwarnings('ignore')

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

model = ChatGroq(
        temperature=0.7,
        model="llama3-70b-8192"
    )

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])




history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL, 
    token=UPSTASH_TOKEN, 
    ttl=500, 
    session_id="chat1"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)

# chain = model | prompt | memory
chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,
    memory=memory
)


while True:
    question = input("You: ")
    if question.lower() == 'exit':
        break
    resp = chain.invoke({ "input": question })
    print(resp["text"])