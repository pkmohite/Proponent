import streamlit as st
import os
import pandas as pd
from assets.code.element_configs import column_config_edit, config_csv_upload, parquet_schema_mf
from assets.code.utils import pass_openAI_key, get_embedding, load_mf_data, verify_password, set_page_config
import pyarrow as pa
import pyarrow.parquet as pq
from streamlit_pdf_viewer import pdf_viewer

## Functions

def add_new_message():
    st.markdown("##### Add New Pain Point")
    cont = st.container(border=True, height=560)
    col1, col2 = cont.columns([3.5, 2])
    
    paint_point = col1.text_area("Enter Customer Pain Point:")
    feature = col1.text_area("Enter Feature Name:")
    value_prop = col1.text_area("Enter Value Proposition:")
    
    tab1, tab2, tab3 = col2.tabs(["Attach PDF", "Attach Video", "Add Web URL"])
    
    # Add a PDF file uploader
    pdf = tab1.file_uploader("Upload Product Slide (PDF):", type=["pdf"])
    if pdf:
        base64_pdf = pdf.getvalue()
        tab1cont = tab1.container(height=300)
        with tab1cont:
            pdf_viewer(input=base64_pdf, width=500)

    # Add a video file uploader
    video = tab2.file_uploader("Upload Product Demo (MP4):", type=["mp4"])
    if video:
        tab2cont = tab2.container()
        with tab2cont:
            st.video(video)
    
    # Add a web URL
    web_url = tab3.text_input("Enter Web URL:")
    # check if the web URL is valid
    if web_url:
        if not web_url.startswith("http"):
            tab3.warning("Please enter a valid URL starting with 'http' or 'https'.")
        else:
            tab3.image(web_url, width=500)
    
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
                [web_url],
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


def manage_mf():
    # Load the data
    mf_data = load_mf_data()
    edited_data = st.data_editor(mf_data, column_config=column_config_edit, use_container_width=True, hide_index=True, num_rows='dynamic')
    
    if st.button("Update Changes"):
        # Delete files from the slides, videos, and web folders if the file name is removed
        for index, row in mf_data.iterrows():
            if row["pdfFile"] is not None and row["pdfFile"] not in edited_data["pdfFile"].values:
                os.remove(os.path.join("slides", row["pdfFile"]))
            if row["videoFile"] is not None and row["videoFile"] not in edited_data["videoFile"].values:
                os.remove(os.path.join("videos", row["videoFile"]))

        # Save the updated data to the Parquet file
        pq.write_table(pa.Table.from_pandas(edited_data, schema=parquet_schema_mf), "assets/mf_embeddings.parquet")

    
def upload_mf_via_csv():
    # Add a download button
    st.markdown("##### Add Painpoints via CSV")
    # Add a download button for the template
    col1, col2 = st.columns([10, 1])
    template_csv = "assets/templates/mf_template.csv"
    with open(template_csv, "rb") as file:
        col2.download_button("Download Template", file, file_name="mf_template.csv", mime="text/csv")
    # Upload the CSV file
    csv_file = col1.file_uploader("PP CSV:", label_visibility= 'collapsed', type=["csv"])
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
    
# Tab 3: Customer Personas
def update_customer_personas():
    st.subheader("Customer Personas")
    st.write("Upload a CSV file with columns category name, persona name, and persona description.")
    # Add a download button for template
    col1, col2 = st.columns([10, 1])
    template_csv = "assets/templates/cp_template.csv"
    with open(template_csv, "r") as file:
        bth = col2.download_button("Download Template", file, file_name="cp_template.csv", mime="text/csv")

    uploaded_file = col1.file_uploader("CP CSV:", label_visibility= 'collapsed', type=["csv"])

    if uploaded_file is not None:
        # Preview the uploaded file using st.write
        df = pd.read_csv(uploaded_file)
        edited_data = st.data_editor(df, hide_index=True, use_container_width=True)
        
        # Add a button to save the uploaded file as a json file
        if st.button("Upload Customer Personas"):
            # Save the edited data as a CSV file
            edited_data.to_csv("assets/customer_personas.csv", index=False)
            st.success("File saved successfully!")

# Setup
def setup_page():
    # Session state setup
    st.session_state.clicked = False
    st.session_state.display_metrics = False
    # Page setup
    set_page_config(page_title="Messaging Manager", page_icon=":speech_balloon:", layout="wide")
    # Password verification
    

# Setup
setup_page()
verify_password()
pass_openAI_key()

## Tabs
st.header("Messaging Manager")
tab1, tab2, tab3, tab4 = st.tabs(["Add New Message", "Modify Messaging Framework", "Upload CSV", "Update Customer Personas"])

# Add painpoint
with tab1:
    add_new_message()

# Modify existing painpoints
with tab2:
    manage_mf()

# Upload CSV
with tab3:
    upload_mf_via_csv()

# Update Customer Personas
with tab4:
    update_customer_personas()