import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
while True:
    query = input("Ask: ")
    response = chat.send_message(query)
    print(response.text)
    