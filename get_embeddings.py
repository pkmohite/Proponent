import json
from openai import OpenAI
from dotenv import load_dotenv
import os
client = OpenAI()

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Load the JSON data from file
with open('messaging_framework.json', 'r') as file:
    data = json.load(file)

# Generate embeddings for each mf
for mf in data:
    customerPainPoint = mf['customerPainPoint']
    featureName = mf['featureName']
    valueProposition = mf['valueProposition']
    embeddings_text = customerPainPoint + ' ' + featureName + ' ' + valueProposition

    embedding = get_embedding(embeddings_text)
    mf['embedding'] = embedding

# Save the updated data with embeddings to a new JSON file
with open('mf_embedded.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Embeddings generated and saved to mf_embedded.json")