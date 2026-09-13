import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chat_history = []

print("Chat started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    response = llm.invoke(chat_history)  # send actual conversation, not a fixed question

    reply_text = response.content[0]['text']
    chat_history.append(AIMessage(content=reply_text))

    print("Gemini:", reply_text, "\n")  # print inside the loop, every turn
    