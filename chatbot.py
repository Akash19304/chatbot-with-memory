from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate 
from langchain.chains.history_aware_retriever import create_history_aware_retriever


def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def create_chain(vectorstore):
    model = ChatGroq(
        temperature=0.7,
        model="llama3-70b-8192"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retreiver = create_history_aware_retriever(
        llm = model,
        retriever=retriever,
        prompt = retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retreiver,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return response['answer']


if __name__ == "__main__":
    load_dotenv()
    os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
    os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
    os.environ['LANGCHAIN_TRACING_V2'] = "true"

    docs = get_document_from_web("https://python.langchain.com/v0.1/docs/modules/chains/")
    vectorstore = create_db(docs)
    chain = create_chain(vectorstore)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant: ", response)