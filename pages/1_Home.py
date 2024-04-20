import os, hmac, json, time
import pandas as pd
import streamlit as st
from assets.code.element_configs import column_config_recommendations, config_about
from assets.code.utils import generate_customized_email, pass_openAI_key, get_embedding, create_env_file, calculate_similarity_ordered, transcribe_video
from assets.code.utils import create_summary, get_themed_logo, update_log_parquet, create_image_deck, displayPDF, verify_password, set_page_config
from moviepy.editor import VideoFileClip, concatenate_videoclips
from assets.code.genHTML import generate_content, generate_feature_section, generate_html_template
import streamlit.components.v1 as components
# from assets.code.utils import create_pdf_deck

## Functions

def click_button():
    st.session_state.clicked = True


def format_display_df(recommendations):
    # Append a select column to the recommendations DataFrame
    recommendations["select"] = [True] * len(recommendations)
    # append normalized similarity score to the data frame, normalized to the min and max similarity score
    recommendations["ss_Normalized"] = (
        recommendations["similarity_score"] - recommendations["similarity_score"].min()
    ) / (
        (recommendations["similarity_score"].max() - recommendations["similarity_score"].min())*1.2
    )
    # append boolean values for PDF, Video files and Web URL
    recommendations["PDF_Present"] = recommendations["pdfFile"].apply(lambda x: bool(x))
    recommendations["Video_Present"] = recommendations["videoFile"].apply(lambda x: bool(x))
    recommendations["Web_URL_Present"] = recommendations["webURL"].apply(lambda x: bool(x))

    return recommendations


def setup_streamlit():
    st.session_state.display_metrics = False
    ## Session State Stuff
    if "clicked" not in st.session_state:
        st.session_state.clicked = False
    if "download_video" not in st.session_state:
        st.session_state.video_download = False
    if "video_text" not in st.session_state:
        st.session_state.video_text = None
    if "chat_text" not in st.session_state:
        st.session_state.chat_text = None
    if "ap_text" not in st.session_state:
        st.session_state.ap_text = None
    if "example_name" not in st.session_state:
        st.session_state.example_name = None
    if "example_title" not in st.session_state:
        st.session_state.example_title = None
    if "example_company" not in st.session_state:
        st.session_state.example_company = None
    # Pass a variable to the set_page_config function
    set_page_config(page_title="Proponent", layout="wide")
    # Verify the password
    verify_password()
    # Set the page logo
    # get_themed_logo()



def get_user_input():
    # Configure the layout
    container_height = 570
    input, persona = st.columns([2, 1])
    
    # 1 - User input
    input.markdown("##### Upload Customer Interaction")
    inputcontainer = input.container(border=True, height=container_height)
    tab1, tab2, tab3 = inputcontainer.tabs(["Upload Video/Audio", "Upload Emai/Chat Text", "Ask Proponent"])

    # 1a - File uploader for video/audio
    video_text = None
    uploadempty = tab1.empty()
    tab1a, tab1b, tab1c = uploadempty.columns([4, 1, 1])
    video_file = tab1a.file_uploader("Upload Video/Audio", type=["mp4", "mov", "avi", "mp3", "wav"], accept_multiple_files=False, label_visibility="collapsed")
    if tab1b.button("Upload Audio/Video File"):
        if video_file:
            with open("downloads/transcribe_cache.mp4", "wb") as file:
                file.write(video_file.read())
            st.session_state.video_text = transcribe_video("downloads/transcribe_cache.mp4")
        else:
            tab1.error("Please upload a video or audio file before proceeding.")
    
    # 1a - Load example video/audio file
    if tab1c.button("Load Audio/Video Example"):
        st.session_state.video_text = "This is a sample transcript of the uploaded video/audio file."
        # save the example video to downloads folder
        example_video = "assets/templates/transcribe_example.mp4"
        with open("downloads/transcribe_cache.mp4", "wb") as file:
            file.write(open(example_video, "rb").read())

    # 1a - Display the video and transcript
    if st.session_state.video_text:
        # uploadempty.empty()
        col1,col2 = tab1.columns([2.5, 1])
        # col1.markdown("##")
        col1.video("downloads/transcribe_cache.mp4")
        video_text = col2.text_area("Transcript", label_visibility="visible", value=st.session_state.video_text,height=container_height-240)

    # 1b - File uploader for chat/transcript
    chat_text = None
    chat1, chat2 = tab2.columns([3, 2])
    chat_file = chat1.file_uploader("Upload Chat/Transcript", type=["txt"], accept_multiple_files=False, label_visibility="collapsed")
    if chat_file:
        st.session_state.chat_text = chat_file.read().decode("utf-8")
    
    # 1b - Load example chat data
    chat_data = pd.read_csv("assets/templates/examples_chat.csv")
    chat_example = chat2.selectbox("Select Example", chat_data["label"].values, index=None, label_visibility="visible")
    if chat_example:
        st.session_state.chat_text = chat_data[chat_data["label"] == chat_example]["text"].values[0]
    
    # 1b - Display the chat transcript
    chat_text = tab2.text_area("Chat Transcript", height= container_height-200, label_visibility="collapsed", placeholder="Upload a chat or transcript file of the customer interaction..",value= st.session_state.chat_text)

    # 1c - Text area for Ask Proponent
    ap_text = None
    ap1, ap2 = tab3.columns([2, 1])
    ap2_container = ap2.container(height=container_height-100, border=False)
    # 1c - Load example ask proponent data
    ap_data = pd.read_csv("assets/templates/examples_text.csv")
    ap_example = ap_data["label"].values
    for i in range(len(ap_example)):
        if ap2_container.button(ap_example[i]):
            st.session_state.ap_text = ap_data["text"].values[i]

    # 1c - Display the ask proponent text
    ap_text = ap1.text_area("Interaction Text", height= container_height-100, label_visibility="collapsed", value = st.session_state.ap_text, placeholder="Describe customer pain point, use-case, feature ask, or any other relevant information..")

    # 1 - Pass the user input to the persona section
    if video_text is not None:
        user_input = video_text
        pass
    elif chat_text is not None:
        user_input = chat_text
        pass
    elif ap_text is not None:
        user_input = ap_text
        pass
    else:
        user_input = None

    # 2 - Load the example profile data
    # Load example profile from assets/templates/hubspot_data.csv
    example_profile = pd.read_csv("assets/templates/hubspot_data.csv")
    # Concatenate the contact_firstname and	contact_lastname columns, then append contact_funtion and company_name into a new dataframe
    example_profile["contact_fullname"] = example_profile["contact_firstname"] + " " + example_profile["contact_lastname"]
    
    # 2a - Column grid for user Customer Name and Company Name
    persona.markdown("##### Personalize (Optional)")
    personacontainer = persona.container(border=True, height=container_height)
    customer_name = personacontainer.selectbox("Customer Name", example_profile["contact_fullname"].tolist(), index=None, placeholder="Select Customer Name")
    row11,row12 = personacontainer.columns([1, 1])
    row21, row22 = personacontainer.columns([1, 1])
    
    example_title, example_company, deal_stage, deal_amount, competitor, history = None, None, None, None, None, None
    if customer_name:
        example_title = example_profile[example_profile["contact_fullname"] == customer_name]["contact_function"].values[0]
        example_company = example_profile[example_profile["contact_fullname"] == customer_name]["company_name"].values[0]
        deal_stage = example_profile[example_profile["contact_fullname"] == customer_name]["deal_stage"].values[0]
        deal_amount = example_profile[example_profile["contact_fullname"] == customer_name]["deal_amount"].values[0]
        competitor = example_profile[example_profile["contact_fullname"] == customer_name]["competitor"].values[0]
        history = example_profile[example_profile["contact_fullname"] == customer_name]["conversation_thread"].values[0]
    customer_title = row11.text_input("Title:", value = example_title)
    customer_company = row12.text_input("Company:", value= example_company)
    deal_stage = row21.text_input("Deal Stage:", value= deal_stage)
    deal_amount = row22.text_input("Deal Amount:", value= deal_amount)
    # competitor = row23.text_input("Competitors:", value= competitor)
    history = personacontainer.text_area("Conversation History", value= history, height=container_height-320)

    # # 2b - Load the customer profiles from assets/customer_profiles.csv
    # row31, row32, row33 = personacontainer.columns([1, 1, 1])
    # customer_profiles = pd.read_csv("assets/customer_personas.csv")
    # category1_value = row31.selectbox("Customer Profile", customer_profiles[customer_profiles["category_name"] == "Buyer Persona"]["persona_name"].tolist(), index=None, placeholder="Buyer Persona")
    # category2_value = row32.selectbox(" ", customer_profiles[customer_profiles["category_name"] == "Company Size"]["persona_name"].tolist(), index=None, placeholder="Company Size", label_visibility="hidden")
    # category3_value = row33.selectbox(" ", customer_profiles[customer_profiles["category_name"] == "Role"]["persona_name"].tolist(), index=None, placeholder="Role", label_visibility="hidden")
    category1_value = None
    category2_value = None
    category3_value = None

    return customer_name, customer_title, customer_company, category1_value, category2_value, category3_value, user_input


def get_recommendations(user_input, customer_name, customer_title, customer_company, category1_value, category2_value, category3_value):
    # Get the summary and recommendations
    summary = create_summary(user_input, customer_name, customer_title, customer_company)
    summary_embedding = get_embedding(user_input) #replace with user_input or summary if simalrity search on summary instead of user input
    df = calculate_similarity_ordered(summary_embedding)
    df_formatted = format_display_df(df)
    top_7 = df_formatted.head(7)
    
    # Store the recommendations in session state
    st.session_state.display_df = top_7
    st.session_state.summary = summary
    
    # Log the recommendations
    top_7_unformatted = df.head(7)
    update_log_parquet(customer_name, customer_title, customer_company, category1_value, category2_value, category3_value, user_input, top_7_unformatted)


def display_recommendations():
    st.markdown("###### Customer Asks:")
    t1container = st.container(border=True, height=370)
    t1container.write(st.session_state.summary)
    st.markdown("###### Recommended Features:")
    selected_df = st.data_editor(
        st.session_state.display_df,
        column_config=column_config_recommendations,
        column_order=["select", "customerPainPoint", "featureName", "valueProposition", "ss_Normalized"],
        hide_index=True,
        use_container_width=True,
    )

    # Store the selected recommendations in a df
    selected_recommendations = selected_df[selected_df["select"] == True]
    
    return selected_recommendations


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


## Setup
setup_streamlit()
create_env_file()
pass_openAI_key()

## Main
# Display the user input fields
customer_name, customer_title, customer_company, category1_value, category2_value, category3_value, user_input = get_user_input()
rec1, rec2, rec3 = st.columns([1.3, 1, 6])

# Button to get recommendations
if rec1.button("Get Recommendations", on_click=click_button):
    # Check if api key is set
    if not os.getenv("USER_API_KEY"):
        st.error("Please set your OpenAI API key in the settings tab before proceeding.")
        st.stop()
    # check if user input is empty
    if not user_input:
        st.error("Please enter a customer interaction text before proceeding.")
        st.stop()

    # Get the recommendations
    get_recommendations(user_input, customer_name, customer_title, customer_company, category1_value, category2_value, category3_value)

if rec2.button("Clear View"):
    # delete st.session_state.clicked
    st.session_state.clicked = False
    st.session_state.video_text = None
    st.session_state.chat_text = None
    st.session_state.ap_text = None
    st.session_state.example_name = None
    st.session_state.example_title = None
    st.session_state.example_company = None
    st.rerun()

# Main content
if st.session_state.clicked:
    st.divider()
    st.markdown("#### Proponent Recommendations:")
    dtab1, dtab2 = st.columns([2, 3])
    # Tab 1 - Customer Asks and Recommendations
    with dtab1:
        selected_recommendations = display_recommendations()

    # Tab 2 - Enablement Center
    with dtab2:
        st.markdown("###### Enablement Center")
        dtab2_cont = st.container(border=True, height=700)
        tab1, tab2, tab3, tab4 = dtab2_cont.tabs(["Draft Email", "Build Sales Deck", "Build Demo Video", "Generate HTML"])

        # Tab 2a - Draft Email
        with tab1:
            # email_body = generate_customized_email(selected_recommendations, user_input, customer_name, customer_title, customer_company)
            email_body = "Email Preview is not available in this demo deployment. Please download the PDF deck and video for the recommendations."
            st.markdown("##### Email Preview:")
            st.markdown(email_body)

        # Tab 2b - Build Sales Deck
        with tab2:
            create_image_deck(selected_recommendations)
            with open("downloads/combined_PDF.pdf", "rb") as file:
                st.download_button(
                    label="Download PDF Deck",
                    data=file.read(),
                    file_name="customized_deck.pdf",
                    mime="application/pdf",
                )
            
            if os.path.exists("downloads/combined_PDF.pdf"):
                displayPDF("downloads/combined_PDF.pdf", st)
            else:
                st.error("Error generating PDF. Please try again or contact me at prashant@yourproponent.com if this persists.")

        # Tab 2c - Build Demo Video
        with tab3:
            # create_video(selected_recommendations) # Uncomment this line in local deployment to enable video generation
            b1, b2 = st.columns([1, 5])
            b2.warning("Video generation is not available in demo. Below preview is pre-generated.") # Comment this line in local deployment
            if os.path.exists("downloads/video.mp4"):
                with open("downloads/video.mp4", "rb") as file:
                    b1.download_button(
                        label="Download MP4 File",
                        data=file.read(),
                        file_name="downloads/video.mp4",
                        mime="video/mp4",
                    )
                st.video("downloads/video.mp4")
            else:
                st.error("Error generating video. Please try again or contact me at prashant@yourproponent.com if this persists.")

        # Tab 2d - Generate HTML
        with tab4:
            # # Generate content for the HTML template using OpenAI
            # hero_title, hero_description, feature_titles, value_propositions, webURL = (
            #     generate_content(
            #         recommendations=selected_recommendations,
            #         user_input=user_input,
            #         customer_name=customer_name,
            #         customer_title=customer_title,
            #         customer_company=customer_company,
            #         model="gpt-3.5-turbo-0125",
            #     )
            # )
            # # Generate HTML for feature sections
            # features = [
            #     generate_feature_section(
            #         feature_titles[i],
            #         value_propositions[i],
            #         webURL[i],
            #     )
            #     for i in range(len(feature_titles))
            # ]
            # deine hero_images
            hero_images = ["https://imagedelivery.net/XawdbiDo2zcR8LA99WkwZA/9ae4b3c7-108b-4635-4d76-489b1d195700/website",
                        "https://dapulse-res.cloudinary.com/image/upload/f_auto,q_auto/remote_mondaycom_static/uploads/NaamaGros/WM-boards/Goals_strategy.png",
                        "https://assets-global.website-files.com/60058af53d79fbd8e14841ea/60181447286c0bee8d42171a_73dc280a-a211-4157-8e7c-b123b1d4ffa0_product_hero_animation_placeholder.png"]

            # # Generate the HTML template
            # html_template = generate_html_template(
            #     hero_title,
            #     hero_description,
            #     hero_images,
            #     features,
            # )
            # # Save the generated HTML template to a file
            # with open("downloads/index.html", "w") as file:
            #     file.write(html_template)

            # View the generated HTML template
            with open("downloads/index.html", "rb") as file:
                st.download_button(
                    label="Download HTML File",
                    data=file.read(),
                    file_name="index.html",
                    mime="text/html",
                )
            with st.container(height=550, border=False):
                
                with open("downloads/index.html", "r") as file:
                    html_template = file.read()
                components.html(html_template, height=4000)
