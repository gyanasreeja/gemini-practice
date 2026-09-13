import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Our "documents" (in real RAG, these come from files/database)
documents = [
    "Ziggy is a full-stack food delivery platform built with FastAPI, PostgreSQL, and ReactJS.",
    "Ziggy uses JWT and RBAC for authentication and authorization.",
    "Docker is used for containerization, and GitHub Actions handles CI/CD for Ziggy.",
    "Gyanasreeja completed her B.Tech in Information Technology in 2025."
]

# Step 2: Convert text into an embedding (a list of numbers representing meaning)
def get_embedding(text, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return np.array(result.embeddings[0].values)

# Embed all documents (as "documents")
doc_embeddings = [get_embedding(doc, task_type="RETRIEVAL_DOCUMENT") for doc in documents]

# Step 3: User's question — embed it as a "query" (different task_type)
question = "What backend framework does Ziggy use?"
question_embedding = get_embedding(question, task_type="RETRIEVAL_QUERY")

# Step 4: Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(question_embedding, doc_emb) for doc_emb in doc_embeddings]

# Step 5: Get top 2 most relevant documents (not just the single best match)
top_n = 2
top_indices = np.argsort(similarities)[-top_n:][::-1]  # highest similarity first

print("Top matches:")
for i in top_indices:
    print(f"  - {documents[i]}  (score: {similarities[i]:.4f})")

best_context = "\n".join([documents[i] for i in top_indices])

# Step 6: Use retrieved context to answer
prompt = f"""
Answer the question using ONLY this context:
{best_context}

Question: {question}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\nAnswer:", response.text)