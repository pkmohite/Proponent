import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfMerger
from fpdf import FPDF
from streamlit_pdf_viewer import pdf_viewer
import base64
from moviepy.editor import VideoFileClip, concatenate_videoclips
from element_configs import column_config_recommendations, config_about

## Functions

def pass_openAI_key(api_key=None):
    if "USER_API_KEY" in os.environ:
        openai.api_key = os.getenv("USER_API_KEY")
    else:
        st.error("OpenAI API key not found. Please set the API key in the Setting page.")


def create_env_file():
    if not os.path.exists(".env"):
        with open(".env", "w") as file:
            file.write("USER_API_KEY=\n")


def get_embedding(text, model="text-embedding-3-large"): # alt is replacing openai with client = OpenAI()
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_customized_email(recommendations, user_input, customer_name, customer_company):

    # Extract the feature names and value propositions from the recommendations DataFrame
    features_str = "\n".join(recommendations["Feature Name"])
    value_prop_str = "\n".join(recommendations["Value Proposition"])

    # Create the conversation for the OpenAI API
    conversation = [
        {
            "role": "system",
            "content": "You are an expert sales communication assistant. Your goal is to craft a personalized, engaging, and concise email under 200 words to follow up with a customer based on their pain points, feature requests, and our proposed solutions.",
        },
        {
            "role": "user",
            "content": f"Here is the context of my conversation with the customer {customer_name} from {customer_company}:\n\n{user_input}\n\nBased on their input, we have identified the following features and value propositions:\n\nFeatures:\n{features_str}\n\nValue Propositions:\n{value_prop_str}\n\nPlease draft a short follow-up email that:\n1. Thanks the customer for their input and acknowledges their pain points\n2. Highlights all the shortlisted features and their corresponding value propositions in a bullet-point format\n3. Explains how these features collectively address their needs and improve their workflow\n4. Ends with a clear call-to-action, inviting them to schedule a demo or discuss further\n\nKeep the email concise, personalized, and focused on the customer's unique situation. Use a friendly yet professional tone.",
        },
        {
            "role": "assistant",
            "content": "Dear [Customer Name],\n\nThank you for taking the time to share your pain points and feature requests with us. We truly appreciate your valuable input and insights.\n\nAfter carefully reviewing your feedback, we believe the following features from our product will comprehensively address your needs:\n\n[List all shortlisted features and their value propositions in bullet points]\n\nTogether, these features will significantly streamline your workflow, increase efficiency, and help you achieve your goals more effectively.\n\nWe would love to show you how our product can be tailored to your specific use case. If you're interested, I would be happy to schedule a personalized demo at your convenience. Please let me know your availability, and I'll set it up.\n\nBest regards,\n[Your Name]\nSales Team \n Generate in Markdown format.",
        },
    ]

    # Generate the email body using the OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview", messages=conversation
    )

    # Extract the generated email body from the API response
    email_body = response.choices[0].message.content

    return email_body


def calculate_similarity_ordered(user_input_embedding, data):
    df = pd.DataFrame()

    for mf in data:
        mf_embedding = mf["embedding"]
        similarity = cosine_similarity(user_input_embedding, mf_embedding)
        mf["similarity_score"] = similarity
        df = df._append(mf, ignore_index=True)

    df.sort_values(by="similarity_score", ascending=False, inplace=True)

    return df


def create_pdf_deck(df):

    # Create a PdfMerger object
    merger = PdfMerger()

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the PDF file name from the "PDF File" column
        pdf_file = row["PDF File Name"]

        # Construct the file path
        file_path = os.path.join("slides", pdf_file)

        # Check if the file exists
        if os.path.exists(file_path):
            # Open the PDF file in read binary mode
            with open(file_path, "rb") as file:
                # Add the PDF file to the merger
                merger.append(file)

    # Specify the output file path
    output_path = "downloads/combined_PDF.pdf"

    # Write the merged PDF to the output file
    with open(output_path, "wb") as file:
        merger.write(file)

    print(f"Combined PDF created: {output_path}")


def create_image_deck(df):
    # Create a list to store the image paths
    image_paths = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the file path
        image_file = row["PDF File Name"]
        file_path = os.path.join("slides", image_file)

        # Check if the file exists
        if os.path.exists(file_path):
            # Add the image path to the list
            image_paths.append(file_path)

    # Specify the output file path
    output_path = "downloads/combined_PDF.pdf"

    # Create a new PDF document with 16:9 layout
    pdf = FPDF(orientation="L", format="Legal")

    # Add each image to the PDF document
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, x=-1, y=2, w=380)

    # Save the PDF document
    pdf.output(output_path)

    print(f"Combined image PDF created: {output_path}")


def create_summary(user_input):
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Extract key customer asks from text I share with you and generate a summary of the customer pain points or asks ONLY. Don't include anything else",
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


def click_button():
    st.session_state.clicked = True


def remove_video():
    os.remove("downloads/video.mp4")


def format_display_df(recommendations):

    # Create a DataFrame for the recommendations with a "Select" column
    recommendations_df = pd.DataFrame(
        {
            "Select": [False] * len(recommendations),
            "Customer Pain Point": recommendations["customerPainPoint"],
            "Feature Name": recommendations["featureName"],
            "Value Proposition": recommendations["valueProposition"],
            # add normalized similarity score to the data frame, normalized to the min and max similarity score
            "Similarity Score": (
                recommendations["similarity_score"]
                - recommendations["similarity_score"].min()
            )
            / (
                recommendations["similarity_score"].max()
                - recommendations["similarity_score"].min()
            ),
            "PDF File": recommendations["pdfFile"].apply(lambda x: bool(x)),
            "Video File": recommendations["videoFile"].apply(lambda x: bool(x)),
            "PDF File Name": recommendations["pdfFile"],
            "Video File Name": recommendations["videoFile"],
        }
    )

    return recommendations_df


def load_data():
    # Load the JSON data from file
    with open("mf_embeddings.json", "r") as file:
        data = json.load(file)
    return data


def create_video(recommendations):
    # Create a list to store the video paths
    video_paths = []

    # Iterate over each row in the DataFrame
    for index, row in recommendations.iterrows():
        # Construct the file path
        video_file = row["Video File Name"]
        file_path = os.path.join("videos", video_file)

        # Check if the file exists
        if os.path.exists(file_path):
            # Add the video path to the list
            video_paths.append(file_path)

    # Concatenate the video clips
    video_clips = [VideoFileClip(video_path) for video_path in video_paths]
    final_clip = concatenate_videoclips(video_clips)

    # Specify the output file path
    output_path = "downloads/video.mp4"

    # Write the concatenated video to the output file
    final_clip.write_videofile(output_path, codec="libx264", fps=24)

    print(f"Concatenated video created: {output_path}")


def displayPDF(file, column = st):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="1000" height="600" type="application/pdf">'

    # Displaying File
    column.markdown(pdf_display, unsafe_allow_html=True)


def setup_streamlit():
    # Pass a variable to the set_page_config function
    my_variable = None
    st.set_page_config(
        page_title="Proponent",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': "mailto:pmohite95@gmail.com",
            'About': config_about,
        }
    )


def load_examples(file_path = "assets/examples.csv"):

    # Load the CSV data into a DataFrame
    data = pd.read_csv(file_path)
    
    # Create a dropdown to select the type
    selected_type = st.sidebar.selectbox("Load Example Conversation", ["None"] + data["type"].unique().tolist())

    # Filter the data based on the selected type
    filtered_data = data[data["type"] == selected_type]

    # Populate the input fields based on the filtered data
    if selected_type != "None":
        customer_name = filtered_data["name"].iloc[0]
        customer_company = filtered_data["company"].iloc[0]
        user_input = filtered_data["text"].iloc[0]
    else:
        customer_name = ""
        customer_company = ""
        user_input = ""
        
    return customer_name, customer_company, user_input

### Streamlit Code

## Session State Stuff
if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "download_video" not in st.session_state:
    st.session_state.video_download = False

## Title and logo
setup_streamlit()
logo, title = st.columns([1, 12])
logo.image("assets/logo.png", width=90)
title.title("Proponent").markdown("# Proponent")

## Set the OpenAI API key
create_env_file()
pass_openAI_key()

## Body Streamlit code

# Load examples
customer_name, customer_company, user_input = load_examples()

# create a column layout
name_col, indus_col = st.columns([1, 1])
name_col.markdown("##### Customer Name:")

customer_name = name_col.text_input("Customer Name:", label_visibility="collapsed", value= customer_name if customer_name else None)
indus_col.markdown("##### Company Name:")
customer_company = indus_col.text_input("Company Name:", label_visibility="collapsed", value= customer_company if customer_company else None)

st.markdown("##### Describe your customer pain point or feature request:")
user_input = st.text_area("Enter your text here:", label_visibility="collapsed", height=400, value= user_input if user_input else None)

# Button to get recommendations
if st.button("Get Recommendations", on_click=click_button):
    data = load_data()
    summary = create_summary(user_input)
    summary_embedding = get_embedding(summary)
    df = calculate_similarity_ordered(summary_embedding, data)
    # top_5 = df.head(5)
    # st.session_state.display_df = format_display_df(top_5)
    df_formatted = format_display_df(df)
    top_7 = df_formatted.head(7)
    st.session_state.display_df = top_7
    st.session_state.summary = summary

if st.session_state.clicked:
    st.divider()
    st.markdown("### Proponent Recommendations:")
    st.markdown("##### Summary of Customer Asks:")
    st.write(st.session_state.summary)
    st.markdown("##### Recommended Features:")
    selected_df = st.data_editor(
        st.session_state.display_df,
        column_config=column_config_recommendations,
        hide_index=True,
        use_container_width=True,
    )

    # Store the selected recommendations in a df
    selected_recommendations = selected_df[selected_df["Select"] == True]

    # Buttons for generating email, sales deck, and video
    st.divider()
    st.markdown("### Sales Enablement Center:")
    col1, col2 = st.columns([1.5,5])

    if col1.button("Draft Custom Email"):
        # Generate a customized email with the recommendations
        email_body = generate_customized_email(
            selected_recommendations, user_input, customer_name, customer_company
        )
        # col2.markdown("##### Email Preview:")
        col2.markdown(email_body)

    # Button to generate customized PDF deck
    if col1.button("Build Sales Deck"):
        # Create a PDF deck with the selected recommendations
        create_image_deck(selected_recommendations)
        with open("downloads/combined_PDF.pdf", "rb") as file:
            col2.download_button(
                label="Download PDF Deck",
                data=file.read(),
                file_name="customized_deck.pdf",
                mime="application/pdf",
            )
        displayPDF("downloads/combined_PDF.pdf", col2)

    # Button to generate a customized video
    if col1.button("Build Demo Video"):
        # Create a video with the selected recommendations
        create_video(selected_recommendations)  ##<<Uncomment this line to generate video
        if os.path.exists("downloads/video.mp4"):
            with open("downloads/video.mp4", "rb") as file:
                col2.download_button(
                    label="Download Video",
                    data=file.read(),
                    file_name="downloads/video.mp4",
                    mime="video/mp4",
                    on_click=remove_video,
                )
            col2.video("downloads/video.mp4")
        else:
            st.error("Error generating video. Please try again.")
