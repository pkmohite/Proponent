import openai
import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfMerger
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from assets.code.element_configs import parquet_schema_log
import assemblyai as aai
from fpdf import FPDF

## OpenAI Functions

## AssemblyAI Functions

def transcribe_video(file):
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    
    transcript = transcriber.transcribe(file)
    # transcript = transcriber.transcribe("./my-local-audio-file.wav")
    
    return transcript.text

# Function to pass the OpenAI key
def pass_openAI_key(api_key=None):
    if "USER_API_KEY" in os.environ:
        openai.api_key = os.getenv("USER_API_KEY")
    else:
        st.sidebar.error("OpenAI API key not found. Please set the API key in the Setting page.")

# Function to get the embedding of a text from the OpenAI API
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding

# Function to generate a summary using the OpenAI API
def create_summary(user_input, customer_name, customer_title, customer_company, model="gpt-3.5-turbo-0125"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f'"You are a helpful sales enablement assistant. I\'m interacting with {customer_name} with title {customer_title} from {customer_company}.\
                    Extract key customer asks from text I share with you and generate a summary of the customer pain points or asks ONLY. Don\'t include anything else"',
            },
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "content": "Here is a summary of your input:\n\n",
            },
        ],
    )
    summary = response.choices[0].message.content
    return summary

# Function to generate a customized email using the OpenAI API
def generate_customized_email(recommendations, user_input, customer_name, customer_title, customer_company, model="gpt-3.5-turbo-0125"):

    # Extract the feature names and value propositions from the recommendations DataFrame
    features_str = "\n".join(recommendations["featureName"])
    value_prop_str = "\n".join(recommendations["valueProposition"])

    # Create the conversation for the OpenAI API
    conversation = [
        {
            "role": "system",
            "content": "You are an expert sales communication assistant. Your goal is to craft a personalized, engaging, and concise email under 200 words to follow up with a customer based on their pain points, feature requests, and our proposed solutions.",
        },
        {
            "role": "user",
            "content": f"Here is the context of my conversation with the customer {customer_name}, {customer_title} from {customer_company}:\n\n{user_input}\n\nBased on their input, we have identified the following features and value propositions:\n\nFeatures:\n{features_str}\n\nValue Propositions:\n{value_prop_str}\n\nPlease draft a short follow-up email that:\n1. Thanks the customer for their input and acknowledges their pain points\n2. Highlights all the shortlisted features and their corresponding value propositions in a bullet-point format\n3. Explains how these features collectively address their needs and improve their workflow\n4. Ends with a clear call-to-action, inviting them to schedule a demo or discuss further\n\nKeep the email concise, personalized, and focused on the customer's unique situation. Use a friendly yet professional tone.",
        },
        {
            "role": "assistant",
            "content": "Dear [Customer Name],\n\nThank you for taking the time to share your pain points and feature requests with us. We truly appreciate your valuable input and insights.\n\nAfter carefully reviewing your feedback, we believe the following features from our product will comprehensively address your needs:\n\n[List all shortlisted features and their value propositions in bullet points]\n\nTogether, these features will significantly streamline your workflow, increase efficiency, and help you achieve your goals more effectively.\n\nWe would love to show you how our product can be tailored to your specific use case. If you're interested, I would be happy to schedule a personalized demo at your convenience. Please let me know your availability, and I'll set it up.\n\nBest regards,\n[Your Name]\nSales Team \n Generate in Markdown format.",
        },
    ]

    # Generate the email body using the OpenAI API
    response = openai.chat.completions.create(
        model=model, messages=conversation
    )

    # Extract the generated email body from the API response
    email_body = response.choices[0].message.content

    return email_body

# Function to find the cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to calculate the similarity scores between the user input and the data
def calculate_similarity_ordered(user_input_embedding):
    
    # Load assets/mf_embeddings.parquet
    mf_data = load_mf_data()
    
    # Calculate the similarity scores
    mf_data["similarity_score"] = mf_data["embedding"].apply(
        lambda x: cosine_similarity(user_input_embedding, x)
    )

    # Sort the data based on the similarity scores
    df = mf_data.sort_values(by="similarity_score", ascending=False)
    
    return df


## Media Processing Functions

# Function to create a PDF deck from the DataFrame
def create_pdf_deck(output_path):

    merger = PdfMerger()

    # Add the start PDF
    merger.append("slides/slide-start.pdf")

    # Add the main content PDF
    merger.append(output_path)

    # Add the end PDF
    merger.append("slides/slide-end.pdf")

    # Write to an output PDF document
    merger.write(output_path)

    merger.close()

    print(f"Combined PDF created: {output_path}")


def create_image_deck(df):
    # Create a list to store the image paths
    image_paths = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the file path
        image_file = row["pdfFile"]
        file_path = os.path.join("slides", image_file)

        # Check if the file exists
        if os.path.exists(file_path):
            # Add the image path to the list
            image_paths.append(file_path)
        else:
            st.error(f"No slides in database. Please upload via Messaging Framework Tab or Contact Support if that doesn't work.")

    # Specify the output file path
    output_path = "downloads/combined_PDF.pdf"

    # Create a new PDF document with 16:9 layout
    pdf = FPDF(orientation="L", format=(285, 510))

    # Add each image to the PDF document
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, x=-1, y=2, w=510)
    
    # Save the PDF document
    pdf.output(output_path)

    create_pdf_deck(output_path)

    print(f"Combined image PDF created: {output_path}")



## DB Functions

# Function to load the data from the Parquet file
def load_mf_data(file="assets/mf_embeddings.parquet"):
    if not os.path.exists(file):  
        st.warning("No data found. Upload painpoints via CSV.")
        st.stop()
    else:
        mf_data = pd.read_parquet(file)
    
    return mf_data


# Function to update the log in a Parquet file
def update_log_parquet(
    customer_name,
    customer_title,
    customer_company,
    persona_category1,
    persona_category2,
    persona_category3,
    user_input,
    top_7,
):
    # Get the current date and time
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
    time = pd.Timestamp.now().strftime("%H:%M:%S")

    # Extract the pain point IDs from the top 7 recommendations
    painPointIDs = [int(id) for id in top_7["painPointId"].tolist()]

    # Specify the Parquet file path
    parquet_file = "assets/log.parquet"

    # Create a new Arrow table with the variable values
    data = [
        [customer_name],
        [customer_title],
        [customer_company],
        [persona_category1],
        [persona_category2],
        [persona_category3],
        [user_input],
        [painPointIDs],
        [date],
        [time],
    ]
    table = pa.Table.from_arrays(data, schema=parquet_schema_log)

    # Check if the Parquet file exists and append the new data
    if os.path.exists(parquet_file):
        # Read the existing Parquet file
        existing_table = pq.read_table(parquet_file)

        # Append the new data to the existing table
        new_table = pa.concat_tables([existing_table, table])

        # Write the updated table to the Parquet file
        pq.write_table(new_table, parquet_file)
    else:
        # Write the table to a new Parquet file
        pq.write_table(table, parquet_file)


# Function to get the log and the mf embeddings
def get_mf_and_log(log_file = 'assets/log.parquet', mf_file = 'assets/mf_embeddings.parquet'):
   
    # Read the Parquet file into an Arrow Table
    table = pq.read_table(log_file)
    
    # Convert the Arrow Table to a Pandas DataFrame
    df = table.to_pandas()
    
    # Convert the 'paintpoints' array into separate columns
    df = pd.concat([df.drop('paintpoints', axis=1), df['paintpoints'].apply(pd.Series).add_prefix('pp')], axis=1)
    
    # Read the data from assets/mf_embeddings.parquet
    mf_content = pq.read_table(mf_file).to_pandas()

    return df, mf_content


## Auxiliary Functions

# Function to create the .env file if it does not exist 
def create_env_file():
    if not os.path.exists(".env"):
        with open(".env", "w") as file:
            file.write("USER_API_KEY=\n")


def get_themed_logo():
    # read assets/themes.csv and check if active theme's vibe column is dark or light
    themes_df = pd.read_csv("assets/themes.csv")
    current_theme_index = int(themes_df[themes_df["active"] == "x"].index[0])
    current_theme_values = themes_df.loc[current_theme_index]
    if current_theme_values["vibe"] == "dark":
        st.image("assets/images/logo_full_white.png", width=300)
    else:
        st.image("assets/images/logo_full_black.png", width=300)
    st.markdown('###')


