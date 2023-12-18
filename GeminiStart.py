import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
"""
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
"""

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Explain Natural Language Processing')

print(response.text)
