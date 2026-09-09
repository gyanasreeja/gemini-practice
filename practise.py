import os
from dotenv import load_dotenv
from google import genai

# load env
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# send prompt
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="who was the first pm of India?"
)
print(response.text)