import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
with open("deepmeaning.jpeg" , "rb") as f:
    image_bytes = f.read()
response = client.models.generate_content(
    model = "gemini-3.6-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
               "what does the picture describe?"
               ]
)    