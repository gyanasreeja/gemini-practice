import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

documents = [
    "Ziggy is a full-stack food delivery platform built with backend as FastAPI,  database as PostgreSQL, and  frontend  with ReactJS.",
    "Ziggy uses JWT and RBAC for authentication and authorization.",
    "Docker is used for containerization, and GitHub Actions handles CI/CD for Ziggy.",
    "Gyanasreeja completed her B.Tech in Information Technology in 2025."
]

vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# New way: build a prompt + document chain, then combine with retriever
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below:\n\n{context}\n\nQuestion: {input}"
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({"input": "What backend framework does Ziggy use?"})
print(result['answer'])
