'''client.chats.create(...) — starts a conversation session (it remembers previous messages)
while True loop — keeps asking you for input until you type "exit"
chat.send_message(...) — sends your message and gets a reply, using the recommended AFC-friendly method'''

'''1. System instruction — tells the model its role/personality before any user message (e.g., "You are a helpful coding tutor" vs "You are a sarcastic assistant")

2. Temperature — controls randomness/creativity (0 = focused & predictable, 1 = more creative/random) '''

import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
#load env
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#start a conversation session
chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a friendly coding tutor helping a beginner learn Python. Keep answers short and simple.",
        temperature=0.7
    )
)
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    response = chat.send_message(user_input)
    print("Gemini:", response.text, "\n")