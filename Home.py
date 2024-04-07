import json
import os
import pandas as pd
import streamlit as st
import base64
from assets.code.element_configs import column_config_recommendations, config_about
from assets.code.utils import pass_openAI_key, get_embedding, create_env_file, calculate_similarity_ordered
from assets.code.utils import create_image_deck, generate_customized_email, create_video, create_summary, get_themed_logo, update_log_parquet
# from assets.code.utils import create_pdf_deck

## Functions

def click_button():
    st.session_state.clicked = True


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


def displayPDF(file, column = st):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="1000" height="600" type="application/pdf">'

    # Displaying File
    column.markdown(pdf_display, unsafe_allow_html=True)


def setup_streamlit():
    ## Session State Stuff
    if "clicked" not in st.session_state:
        st.session_state.clicked = False
    if "download_video" not in st.session_state:
        st.session_state.video_download = False
    
    # Pass a variable to the set_page_config function
    st.set_page_config(
        page_title="Proponent", page_icon=None, layout="wide", 
        initial_sidebar_state="expanded",
        menu_items={'Get Help': "mailto:prashant@yourproponent.com",
                    'About': config_about})
    
    # Set the page logo
    get_themed_logo()


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
        customer_title = filtered_data["title"].iloc[0]
        customer_company = filtered_data["company"].iloc[0]
        user_input = filtered_data["text"].iloc[0]
    else:
        customer_name = ""
        customer_title = ""
        customer_company = ""
        user_input = ""
        
    return customer_name, customer_title, customer_company, user_input


def get_user_input():
    # Load examples
    customer_name, customer_title, customer_company, user_input = load_examples()

    # Column grid for user Customer Name and Company Name
    name_col, title_col, indus_col = st.columns([1, 1, 1])
    name_col.markdown("##### Name")
    customer_name = name_col.text_input("Customer Name:", label_visibility="collapsed", value= customer_name if customer_name else None)
    title_col.markdown("##### Title")
    customer_title = title_col.text_input("Customer Title:", label_visibility="collapsed", value= customer_title if customer_title else None)
    indus_col.markdown("##### Company")
    customer_company = indus_col.text_input("Company Name:", label_visibility="collapsed", value= customer_company if customer_company else None)

    # Load the customer profiles from assets/customer_profiles.json
    with open("assets/customer_personas.json", "r") as file:
        customer_profiles = json.load(file)

    # Create a selectbox for each category
    st.markdown("##### Customer Persona")
    cp1, cp2, cp3 = st.columns([1, 1, 1])
    category1_value = cp1.selectbox("cat1", [persona["persona name"] for persona in customer_profiles[list(customer_profiles.keys())[0]]], index = None, placeholder = list(customer_profiles.keys())[0], label_visibility='collapsed')
    category2_value = cp2.selectbox("cat2", [persona["persona name"] for persona in customer_profiles[list(customer_profiles.keys())[1]]], index = None, placeholder = list(customer_profiles.keys())[1], label_visibility='collapsed')
    category3_value = cp3.selectbox("cat3", [persona["persona name"] for persona in customer_profiles[list(customer_profiles.keys())[2]]], index = None, placeholder = list(customer_profiles.keys())[2], label_visibility='collapsed')

    # Text area for user input
    st.markdown("##### Customer Interaction Text")
    user_input = st.text_area("Interaction Text", label_visibility="collapsed", height=280, placeholder="Enter the transcript of your customer interaction",value= user_input if user_input else None)

    return customer_name, customer_title, customer_company, category1_value, category2_value, category3_value, user_input

def display_recommendations():
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
    
    return selected_recommendations

## Setup
setup_streamlit()
create_env_file()
pass_openAI_key()

## Main
# Display the user input fields
customer_name, customer_title, customer_company, category1_value, category2_value, category3_value, user_input = get_user_input()

# Button to get recommendations
if st.button("Get Recommendations", on_click=click_button):

    def get_recommendations(user_input, customer_name, customer_title, customer_company):
        # Get the summary and recommendations
        summary = create_summary(user_input, customer_name, customer_title, customer_company)
        summary_embedding = get_embedding(summary)
        df = calculate_similarity_ordered(summary_embedding)
        df_formatted = format_display_df(df)
        top_7 = df_formatted.head(7)
        
        # Store the recommendations in session state
        st.session_state.display_df = top_7
        st.session_state.summary = summary
    # Log the recommendations
    top_7_unformatted = df.head(7)
    update_log_parquet(customer_name, customer_title, customer_company, category1_value, category2_value, category1_value, user_input, top_7_unformatted)

if st.session_state.clicked:
    # Display the recommendations
    st.divider()
    selected_recommendations = display_recommendations()
    
    # Sales Enablement Center
    st.divider()
    st.markdown("### Sales Enablement Center:")
    col1, col2 = st.columns([1.5,5])

    if col1.button("Draft Custom Email"):
        # Generate a customized email with the recommendations
        email_body = generate_customized_email(
            selected_recommendations, user_input, customer_name, customer_title, customer_company
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
        if not os.path.exists("downloads/video.mp4"):
            create_video(selected_recommendations)
        
        if os.path.exists("downloads/video.mp4"):
            with open("downloads/video.mp4", "rb") as file:
                col2.download_button(
                    label="Download Video",
                    data=file.read(),
                    file_name="downloads/video.mp4",
                    mime="video/mp4",
                    on_click=os.remove("downloads/video.mp4"),
                )
            col2.video("downloads/video.mp4")
        else:
            st.error("Error generating video. Please try again.")
