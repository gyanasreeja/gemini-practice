import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
output_parser = StrOutputParser()

# LCEL: chain components together with |
chain = prompt | llm | output_parser

print("Streaming Chat! Type 'exit' to quit.\n")

while True:
    user_input = input("Topic: ")
    if user_input.lower() == "exit":
        break

    try:
        print("Gemini: ", end="", flush=True)
        for chunk in chain.stream({"topic": user_input}):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print("Something went wrong:", e, "\n")
        