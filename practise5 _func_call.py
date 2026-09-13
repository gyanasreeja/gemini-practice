import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Define a real Python function
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

# Step 2: Give Gemini access to it
chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        tools=[add_numbers]
    )
)

print("Chat started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    try:
        response = chat.send_message(user_input)
        print("Gemini:", response.text, "\n")
    except Exception as e:
        print("Something went wrong:", e, "\n")