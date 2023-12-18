import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
import PIL.Image

img = PIL.Image.open('Image1.jpg')

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(["Explain this image", img], stream=True)
response.resolve()

print(response.text)
