import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Step 1: Define a reusable template with placeholders
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer this question about {topic} in {style} style: {question}"
)

# Step 2: Fill in the placeholders
formatted_prompt = prompt.format_messages(
    topic="Python",
    style="simple and beginner-friendly",
    question="What is a list?"
)

# Step 3: Send it to the model
response = llm.invoke(formatted_prompt)
print(response.content[0]['text'])