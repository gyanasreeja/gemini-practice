import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a friendly coding tutor helping a beginner learn Python. Keep answers short and simple.",
        temperature=0.7
    )
)

print("Chat started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    try:
        print("Gemini: ", end="", flush=True)
        for chunk in chat.send_message_stream(user_input):
            print(chunk.text, end="", flush=True)
        print("\n")

    except Exception as e:
        print("Something went wrong:", e, "\n")
        