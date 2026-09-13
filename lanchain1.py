import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Create the LangChain model wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Send a simple message
response = llm.invoke("Explain what LangChain is in 2 simple sentences.")
print(response.content[0]['text'])
