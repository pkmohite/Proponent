import csv
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

client = OpenAI()


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def csv_to_json(csv_file):
    # Read the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        # Convert each row to a dictionary and store in a list
        data = [row for row in reader]

    return data

# Specify the input CSV file path and output JSON file path
mf_csv = "mf_content.csv"
embeddings_json = "mf_embeddings.json"

# Call the function to convert CSV to JSON
data = csv_to_json(mf_csv)

# Generate embeddings for each mf
for mf in data:
    customerPainPoint = mf["customerPainPoint"]
    featureName = mf["featureName"]
    valueProposition = mf["valueProposition"]
    embeddings_text = customerPainPoint + " " + featureName + " " + valueProposition

    embedding = get_embedding(embeddings_text)
    mf["embedding"] = embedding

# if embedding generation is successful
if data:
    # Save the updated data with embeddings to a new JSON file
    with open(embeddings_json, "w") as file:
        json.dump(data, file, indent=4)
    
    print("Embedding generation and file saving successful!")
