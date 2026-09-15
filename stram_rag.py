import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

documents = [
    "Ziggy is a full-stack food delivery platform. Backend: FastAPI (Python) with PostgreSQL database. Frontend: ReactJS. Auth: JWT and RBAC. DevOps: Docker, GitHub Actions for CI/CD.",
    "Gyanasreeja completed her B.Tech in Information Technology in 2025."
]

vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer based on this context:\n{context}\n\nQuestion: {question}"
)

# Helper: turn retrieved documents into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What backend framework does Ziggy use?"
answer = rag_chain.invoke(question)
print(answer)
