import streamlit as st
import os
import pandas as pd
from assets.code.element_configs import column_config_edit, config_csv_upload, parquet_schema_mf
from assets.code.utils import pass_openAI_key, get_embedding, load_mf_data
import pyarrow as pa
import pyarrow.parquet as pq


## Functions

## Streamlit Functions
def add_new_message():
    st.markdown("#### Add New Message")
    paint_point = st.text_input("Enter Customer Pain Point:")
    feature = st.text_input("Enter Feature Name:")
    value_prop = st.text_input("Enter Value Proposition:")
    pdf = st.file_uploader("Upload Product Slide (PDF):", type=["pdf"])
    video = st.file_uploader("Upload Product Demo (MP4):", type=["mp4"])

    if st.button("Add"):
        if pdf is not None or video is not None:
            
            # Load the existing data
            mf_data = load_mf_data()

            # Get embeddings for the new painpoint
            embeddings_text = paint_point + " " + feature + " " + value_prop
            embedding = get_embedding(embeddings_text)
            
            # Create a new Arrow table with the variable values
            data = [
                [mf_data["painPointId"].max() + 1],
                [paint_point],
                [feature],
                [value_prop],
                [pdf.name if pdf else None],
                [video.name if video else None],
                [embedding],
            ]
            table = pa.Table.from_arrays(data, schema=parquet_schema_mf)

            # Check if the Parquet file exists and append the new data
            if os.path.exists("assets/mf_embeddings.parquet"):
                # Read the existing Parquet file
                existing_table = pq.read_table("assets/mf_embeddings.parquet")

                # Append the new data to the existing table
                new_table = pa.concat_tables([existing_table, table])

                # Write the updated table to the Parquet file
                pq.write_table(new_table, "assets/mf_embeddings.parquet")
            else:
                # Write the table to a new Parquet file
                pq.write_table(table, "assets/mf_embeddings.parquet")

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


def edit_message():
    # Load the data
    mf_data = load_mf_data()

    # Create a new list to store the formatted data
    editor_data = []
    for mf in mf_data:
        editor_mf = {
            #"selected": False,
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
        editor_data, column_config=column_config_edit, hide_index=True, use_container_width=True, num_rows='dynamic'
    )
    
    # # Get the selected painpoints
    # selected_painpoints = [mf for mf in edited_data if mf["selected"] == True]
    
    # col1, col2, col3 = st.columns([1, 1, 5])
    # if col1.button("Save Selected Records"):
    #     # Save implementation is pending
    #     st.success("Save implementation is pending!")
        
    # if col2.button("Delete Selected Records"):
    #     for painpoint in selected_painpoints:
    #         # Delete the painpoint from the data
    #         delete_painpoint_from_content(painpoint)
    #         delete_painpoint_from_embeddings(painpoint, data)
    #     st.success("Painpoints deleted successfully!")


def manage_mf():
    # Load the data
    mf_data = load_mf_data()
    edited_data = st.data_editor(mf_data, column_config=column_config_edit, use_container_width=True, hide_index=True, num_rows='dynamic')
    
    if st.button("Update Changes"):
        
        # Save the updated data to the Parquet file
        pq.write_table(pa.Table.from_pandas(edited_data, schema=parquet_schema_mf), "assets/mf_embeddings.parquet")

    
def upload_mf_via_csv():
    
    st.markdown("#### Add Painpoints via CSV")
    csv_file = st.file_uploader("Upload CSV File:", type=["csv"])
    if csv_file is not None:
        # Display the data editor
        csv_display = pd.read_csv(csv_file)
        edited_data = st.data_editor(csv_display, column_config=config_csv_upload, use_container_width=True, hide_index=True, num_rows='dynamic')
        
        # Create Messaging records from CSV
        if st.button("Upload CSV"):
            # Create a new DataFrame to store the data with embeddings
            data_with_embeddings = edited_data.copy()
            data_with_embeddings["embedding"] = None
            
            # Check if the CSV file has the required columns
            required_columns = ["customerPainPoint", "featureName", "valueProposition"]
            if all(column in edited_data.columns for column in required_columns):
                # Get embeddings for the new painpoints and add them as a new column to the data
                for index, row in edited_data.iterrows():
                    embeddings_text = (
                        row["customerPainPoint"] + " " + row["featureName"] + " " + row["valueProposition"]
                    )
                    embedding = get_embedding(embeddings_text)
                    data_with_embeddings.at[index, "embedding"] = embedding

                # Update the mf_embeddings.parquet file
                update_mf_parquet(data_with_embeddings)
                st.success("Painpoints added successfully!")
            else:
                st.error("Please make sure the CSV file has customerPainPoint, featureName, and valueProposition columns.")


def update_mf_parquet(data):

    # Convert the data to a PyArrow table
    table = pa.Table.from_pandas(data, schema=parquet_schema_mf)
    parquet_file = "assets/mf_embeddings.parquet"

    if os.path.exists(parquet_file):
        # Read the existing parquet file
        existing_table = pq.read_table(parquet_file)
        
        # Append the new data to the existing table
        new_table = pa.concat_tables([existing_table, table])

        # Write the updated table to the Parquet file
        pq.write_table(new_table, parquet_file)   

    else:        
        # Write the table to a new Parquet file
        pq.write_table(table, parquet_file)
    

# Setup
st.set_page_config(page_title="Messaging Manager", page_icon=":speech_balloon:", layout="wide")
st.title("Messaging Manager")

## Pass OpenAI API key
pass_openAI_key()

## Tabs
tab1, tab2, tab3 = st.tabs(["Add New Message", "Modify Messaging Framework", "Upload CSV"])

# Add painpoint
with tab1:
    add_new_message()

# Modify existing painpoints
with tab2:
    #edit_message(data)
    manage_mf()

# Upload CSV
with tab3:
    upload_mf_via_csv()

