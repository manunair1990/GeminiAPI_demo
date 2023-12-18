import google.generativeai as genai
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
title = "The next generation of AI for developers and Google Workspace"
model = 'models/embedding-001'

DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console. Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
DOCUMENT2 = "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."
DOCUMENT3 = "Shifting Gears  Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

texts = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
df = pd.DataFrame(texts)
df.columns = ['Text']
print(df)
# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(text):
  # title field is optional.
  return genai.embed_content(model=model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

df['Embeddings'] = df['Text'].apply(embed_fn)
#print(df)

query = "How do you shift gears in the Google car?"
print(query)
#model = 'models/embedding-001'

request = genai.embed_content(model=model,
                              content=query,
                              task_type="retrieval_query")

def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Text'] # Return text from index with max value

passage = find_best_passage(query, df)
print(passage)