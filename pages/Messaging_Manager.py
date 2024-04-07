import streamlit as st
import json
import os
import csv
from openai import OpenAI
import openai
from dotenv import load_dotenv
import pandas as pd
from assets.code.element_configs import column_config_edit, config_csv_upload

#client = OpenAI()
def pass_openAI_key(openai_key = None):
    if "USER_API_KEY" in os.environ:
        openai.api_key = os.getenv("USER_API_KEY")
    else:
        st.error("OpenAI API key not found. Please set the API key in the Setting page.")


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding


def load_csv(csv_file="mf_content.csv"):
    # Read the CSV file
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        # Convert each row to a dictionary and store in a list
        data = [row for row in reader]

    return data


def load_json(json_file="mf_embeddings.json"):
    # Read the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    return data


def add_painpoint_to_content(painpoint):
    # Read the existing data from the CSV file
    data = load_csv()
    # Add the new painpoint to the data
    data.append(painpoint)
    # Save the data back to the CSV file
    with open("mf_content.csv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print("Content CSV saved successfully!")


def add_painpoint_to_embeddings(painpoint, data):

    # Generate the embedding for the new painpoint
    embeddings_text = (
        painpoint["customerPainPoint"]
        + " "
        + painpoint["featureName"]
        + " "
        + painpoint["valueProposition"]
    )
    embedding = get_embedding(embeddings_text)
    painpoint["embedding"] = embedding
    data.append(painpoint)

    # Save the updated data with embeddings to a new JSON file
    with open("mf_embeddings.json", "w") as file:
        json.dump(data, file, indent=4)
    print("Embedding generation and JSON file saving successful!")


def delete_painpoint_from_content(painpoint):
    # Read the existing data from the CSV file
    data = load_csv()
    # Delete the painpoint from the data
    data.remove(painpoint)
    # Save the data back to the CSV file
    with open("mf_content.csv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def delete_painpoint_from_embeddings(painpoint, data):
    # Delete the painpoint from the data
    data.remove(painpoint)
    # Save the updated data with embeddings to a new JSON file
    with open("mf_embeddings.json", "w") as file:
        json.dump(data, file, indent=4)


## Streamlit Functions
    
def add_new_message(data):
    st.markdown("#### Add New Message")
    paint_point = st.text_input("Enter Customer Pain Point:")
    feature = st.text_input("Enter Feature Name:")
    value_prop = st.text_input("Enter Value Proposition:")
    pdf = st.file_uploader("Upload Product Slide (PDF):", type=["pdf"])
    video = st.file_uploader("Upload Product Demo (MP4):", type=["mp4"])

    if st.button("Add"):
        if pdf is not None or video is not None:
            new_painpoint = {
                "painPointId": str(int(data[-1]["painPointId"]) + 1),
                "customerPainPoint": paint_point,
                "featureName": feature,
                "valueProposition": value_prop,
                "pdfFile": pdf.name if pdf is not None else "",
                "videoFile": video.name if video is not None else "",
            }

            # Add the new painpoint to the data
            add_painpoint_to_content(new_painpoint)

            # Add the new painpoint to mf_embeddings.json
            add_painpoint_to_embeddings(new_painpoint, data)

            # Save the pdf file in the slides folder
            if pdf is not None:
                with open(os.path.join("slides", pdf.name), "wb") as file:
                    file.write(pdf.getbuffer())

            # Save the video file in the videos folder
            if video is not None:
                with open(os.path.join("videos", video.name), "wb") as file:
                    file.write(video.getbuffer())

            st.success("Painpoint added successfully!")
        else:
            st.error("Please attach a PDF or MP4 file to save the painpoint.")


def edit_message(data):
    # Create a new list to store the formatted data
    editor_data = []
    for mf in data:
        editor_mf = {
            "selected": False,
            "painPointId": mf["painPointId"],
            "customerPainPoint": mf["customerPainPoint"],
            "featureName": mf["featureName"],
            "valueProposition": mf["valueProposition"],
            "pdfFile": mf["pdfFile"],
            "videoFile": mf["videoFile"],
            "checkPDF": bool(mf["pdfFile"]),
            "checkVideo": bool(mf["videoFile"]),
        }
        editor_data.append(editor_mf)
    
    # Display the data editor
    st.markdown("#### Modify Existing Pain Points")
    edited_data = st.data_editor(
        editor_data, column_config=column_config_edit, hide_index=True, use_container_width=True
    )
    
    # Get the selected painpoints
    selected_painpoints = [mf for mf in edited_data if mf["selected"] == True]
    
    col1, col2, col3 = st.columns([1, 1, 5])
    if col1.button("Save Selected Records"):
        # Save implementation is pending
        st.success("Save implementation is pending!")
        
    if col2.button("Delete Selected Records"):
        for painpoint in selected_painpoints:
            # Delete the painpoint from the data
            delete_painpoint_from_content(painpoint)
            delete_painpoint_from_embeddings(painpoint, data)
        st.success("Painpoints deleted successfully!")


def upload_message_via_csv():    
    
    st.markdown("#### Add Painpoints via CSV")
    csv_file = st.file_uploader("Upload CSV File:", type=["csv"])
    if csv_file is not None:
        csv_display = pd.read_csv(csv_file)

        # View the data
        st.data_editor(csv_display, column_config=config_csv_upload, use_container_width=True)
        if st.button("Upload CSV"):

            # Check if the CSV file has the required columns
            required_columns = ["customerPainPoint", "featureName", "valueProposition"]
            if all(column in data[0] for column in required_columns):
                for row in data:
                    # Add the new painpoint to the data
                    add_painpoint_to_content(row)

                    # Add the new painpoint to mf_embeddings.json
                    add_painpoint_to_embeddings(row, data)

                    # Save the pdf file in the slides folder
                    if row["pdfFile"]:
                        with open(os.path.join("slides", row["pdfFile"]), "wb") as file:
                            file.write(row["pdfFile"].getbuffer())

                    # Save the video file in the videos folder
                    if row["videoFile"]:
                        with open(os.path.join("videos", row["videoFile"]), "wb") as file:
                            file.write(row["videoFile"].getbuffer())

                st.success("Painpoints added successfully!")
            else:
                st.error(
                    f"Please make sure the CSV file has the required columns: {', '.join(required_columns)} and one of pdfFile or videoFile."
                )


# Setup
st.set_page_config(page_title="Messaging Manager", page_icon=":speech_balloon:", layout="wide")
st.title("Messaging Manager")

# Load the data
data = load_json()

## Pass OpenAI API key
pass_openAI_key()

## Tabs
tab1, tab2, tab3 = st.tabs(["Add New Message", "Modify Messaging Framework", "Upload CSV"])

# Add painpoint
with tab1:
    add_new_message(data)

# Modify existing painpoints
with tab2:
    edit_message(data)
        
# Upload CSV
with tab3:
    upload_message_via_csv()

