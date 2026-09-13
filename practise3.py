import os
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a friendly coding tutor helping a beginner learn Python. Keep answers short and simple.",
        temperature=0.5
    )
)

print("Chat started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    try:
        response = chat.send_message(user_input)
        print("Gemini:", response.text)

        usage = response.usage_metadata
        print(f"[Tokens -> input: {usage.prompt_token_count}, output: {usage.candidates_token_count}, total: {usage.total_token_count}]\n")
    except Exception as e:
        print("Something went wrong:", e, "\n")