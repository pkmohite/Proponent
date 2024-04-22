import os, hmac, json, time
import pandas as pd
import streamlit as st
from assets.code.element_configs import column_config_recommendations, config_about
from assets.code.utils import generate_customized_email, pass_openAI_key, get_embedding, calculate_similarity_ordered, transcribe_video
from assets.code.utils import create_summary, update_log_parquet, create_image_deck, displayPDF, verify_password, set_page_config, generate_enyk
from moviepy.editor import VideoFileClip, concatenate_videoclips
from assets.code.genHTML import generate_content, generate_feature_section, generate_html_template
import streamlit.components.v1 as components
# from assets.code.utils import create_pdf_deck

## Functions

def click_get_recc():
    st.session_state.clicked = True


def format_display_df(recommendations):
    # Mark first 5 recommendations as selected by default
    recommendations["select"] = [True] * 5 + [False] * (len(recommendations) - 5)
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
    # Pass a variable to the set_page_config function
    set_page_config(page_title="Proponent", layout="wide")
    # Verify the password
    verify_password()
    # Set the page logo
    # get_themed_logo()


def create_video(recommendations):
    # Create a list to store the video paths
    video_paths = []

    # Iterate over each row in the DataFrame
    for index, row in recommendations.iterrows():
        # Construct the file path
        video_file = row["videoFile"]
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


def get_video_input(container_height):
    # 1a - File uploader for video/audio
    video_text = None
    tab1a, tab1b, tab1c = st.columns([4, 1, 1])
    video_file = tab1a.file_uploader("Upload Video/Audio", type=["mp4", "mov", "avi", "mp3", "wav"], accept_multiple_files=False, label_visibility="collapsed")
    if tab1b.button("Upload Audio/Video File"):
        if video_file:
            with open("downloads/transcribe_cache.mp4", "wb") as file:
                file.write(video_file.read())
            st.session_state.video_text = transcribe_video("downloads/transcribe_cache.mp4")
        else:
            st.error("Please upload a video or audio file before proceeding.")
    
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
        col1,col2 = st.columns([2.5, 1])
        col1.markdown("#")
        col1.video("downloads/transcribe_cache.mp4")
        video_text = col2.text_area("Transcript", label_visibility="visible", value=st.session_state.video_text,height=container_height-250)

    return video_text


def get_text_input(container_height):
    # 1b - File uploader for chat/transcript
    chat_text = None
    chat1, chat2 = st.columns([3, 2])
    chat_file = chat1.file_uploader("Upload Chat/Transcript", type=["txt"], accept_multiple_files=False, label_visibility="collapsed")
    if chat_file:
        st.session_state.chat_text = chat_file.read().decode("utf-8")
    
    # 1b - Load example chat data
    chat_data = pd.read_csv("assets/templates/examples_chat.csv")
    chat_example = chat2.selectbox("Select Example", chat_data["label"].values, index=None, label_visibility="visible")
    if chat_example:
        st.session_state.chat_text = chat_data[chat_data["label"] == chat_example]["text"].values[0]
    
    # 1b - Display the chat transcript
    chat_text = st.text_area("Chat Transcript", height= container_height-200, label_visibility="collapsed", placeholder="Upload a chat or transcript file of the customer interaction..",value= st.session_state.chat_text)

    return chat_text


def get_ap_input(container_height):
    ap_text = None
    ap1, ap2 = st.columns([2, 1])
    ap2_container = ap2.container(height=container_height-100, border=False)
    # 1c - Load example ask proponent data
    ap_data = pd.read_csv("assets/templates/examples_text.csv")
    ap_example = ap_data["label"].values
    for i in range(len(ap_example)):
        if ap2_container.button(ap_example[i]):
            st.session_state.ap_text = ap_data["text"].values[i]

    # 1c - Display the ask proponent text
    ap_text = ap1.text_area("Interaction Text", height= container_height-100, label_visibility="collapsed", value = st.session_state.ap_text, placeholder="Describe customer pain point, use-case, feature ask, or any other relevant information..")

    return ap_text


def get_customer_profile(personacontainer, container_height):
    # Load the example profile data
    example_profile = pd.read_csv("assets/templates/hubspot_data.csv")
    example_profile["contact_fullname"] = example_profile["contact_firstname"] + " " + example_profile["contact_lastname"]

    # Fetch logic for customer name, title, company, deal stage, deal amount, competitor, history
    customer_name = personacontainer.selectbox("Customer Name", example_profile["contact_fullname"].tolist(), index=None, placeholder="Select Customer Name")
    row11,row12 = personacontainer.columns([1, 1])
    row21, row22 = personacontainer.columns([1, 1])
    # row31, row32, row33 = personacontainer.columns([1, 1, 1])
    
    customer_title, customer_company, deal_stage, deal_amount, competitor, history = None, None, None, None, None, None
    if customer_name:
        customer_title = example_profile[example_profile["contact_fullname"] == customer_name]["contact_function"].values[0]
        customer_company = example_profile[example_profile["contact_fullname"] == customer_name]["company_name"].values[0]
        deal_stage = example_profile[example_profile["contact_fullname"] == customer_name]["deal_stage"].values[0]
        # deal_amount = example_profile[example_profile["contact_fullname"] == customer_name]["deal_amount"].values[0]
        competitor = example_profile[example_profile["contact_fullname"] == customer_name]["competitor"].values[0]
        history = example_profile[example_profile["contact_fullname"] == customer_name]["conversation_thread"].values[0]
    
    # Input fields for customer name, title, company, deal stage, deal amount, competitor, history
    customer_title = row11.text_input("Title:", value = customer_title, key="title")
    customer_company = row12.text_input("Company:", value= customer_company, key="company")
    deal_stage = row21.text_input("Deal Stage:", value= deal_stage, key="deal_stage")
    # deal_amount = row21.text_input("Deal Amount:", value= deal_amount, key="deal_amount")
    competitor = row22.text_input("Competitor:", value= competitor, key="competitor")

    # # Prepare the customer profile data
    # customer_profiles = pd.read_csv("assets/customer_personas.csv")
    # cp1 = row31.selectbox("Customer Profile", customer_profiles[customer_profiles["category_name"] == "Buyer Persona"]["persona_name"].tolist(), index=None, placeholder="Buyer Persona")
    # cp2 = row32.selectbox(" ", customer_profiles[customer_profiles["category_name"] == "Company Size"]["persona_name"].tolist(), index=None, placeholder="Company Size", label_visibility="hidden")
    # cp3 = row33.selectbox(" ", customer_profiles[customer_profiles["category_name"] == "Role"]["persona_name"].tolist(), index=None, placeholder="Role", label_visibility="hidden")
    cp1,cp2,cp3 = None, None, None
    # Text area for conversation history
    history = personacontainer.text_area("Conversation History", value= history, height=container_height-320, key="history")

    return customer_name, customer_title, customer_company, deal_stage, deal_amount, competitor, history, cp1, cp2, cp3


def get_user_input(container_height = 580):
    # Configure the layout
    input, persona = st.columns([2, 1])
    # persona, input = st.columns([1, 2])
    # 1 - User input
    input.markdown("##### Upload Customer Interaction")
    inputcontainer = input.container(border=True, height=container_height)
    tab1, tab2, tab3 = inputcontainer.tabs(["Upload Video/Audio", "Upload Emai/Chat Text", "Ask Proponent"])

    # 1a - File uploader for video/audio
    with tab1:
        video_text = get_video_input(container_height)

    # 1b - File uploader for chat/transcript
    with tab2:
        chat_text = get_text_input(container_height)

    # 1c - Text area for Ask Proponent
    with tab3:
        ap_text = get_ap_input(container_height)

    # Pass the user input to the persona section
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
    st.session_state.user_input = user_input

    # 2 - Load the example profile data
    persona.markdown("##### Select Customer")
    personacontainer = persona.container(border=True, height=container_height)

    with personacontainer:
        (
            customer_name,
            customer_title,
            customer_company,
            deal_stage,
            deal_amount,
            competitor,
            history,
            cp1,
            cp2,
            cp3,
        ) = get_customer_profile(personacontainer, container_height)

    # 3 - User input buttons
    rec1, rec2, rec3 = st.columns([1.3, 1, 6])

    # Button to get recommendations
    if rec1.button("Get Recommendations", on_click=click_get_recc):
        # Check if api key is set
        if not os.getenv("USER_API_KEY"):
            st.error("OpenAI Authentication error. Please try again or contact me at prashant@yourproponent.com")
            st.stop()
        # check if user input is empty
        if not user_input:
            st.error("Please enter a customer interaction text before proceeding.")
            st.stop()

        # Get the recommendations
        get_recommendations(user_input, customer_name, customer_title, customer_company, cp1, cp2, cp3, history, competitor)

    # Button to clear the view
    if rec2.button("Clear View"):
        # delete st.session_state.clicked
        st.session_state.clicked = False
        st.session_state.video_text = None
        st.session_state.chat_text = None
        st.session_state.ap_text = None
        st.rerun()


    return customer_name, customer_title, customer_company, cp1, cp2, cp3, user_input, history, competitor


def get_recommendations(user_input, customer_name, customer_title, customer_company, cp1, cp2, cp3, history, competitor):
    # Get the summary and recommendations
    # summary = create_summary(user_input, customer_name, customer_title, customer_company)
    summary_embedding = get_embedding(user_input) #replace with user_input or summary if simalrity search on summary instead of user input
    df = calculate_similarity_ordered(summary_embedding)
    df_formatted = format_display_df(df)
    
    # Store the recommendations in session state
    st.session_state.display_df = df_formatted.head(7)
    # st.session_state.summary = summary
    st.session_state.enyk = generate_enyk(df_formatted.head(7), user_input, customer_name, customer_title, customer_company, history, competitor)
    
    # Log the recommendations
    update_log_parquet(customer_name, customer_title, customer_company, cp1, cp2, cp3, user_input, df.head(7))


def dislay_enyk():
    enyk = st.session_state.enyk
    # enyk = "Everything You Need to Know is not available in this demo deployment. Please download the PDF deck and video for the recommendations."
    # st.write(enyk)
    with st.container(border=True, height=485):
        tab1, tab2 = st.tabs(["Interaction Summary", "Historical Needs"])
        ct1 = tab1.container(height=395, border=False)
        ct1.markdown(enyk["customer_needs_summary"])
        # ct1.markdown(st.session_state.summary)
        ct2 = tab2.container(height=395, border=False)
        ct2.markdown(enyk["historical_needs_summary"])
    
    with st.container(border=True, height=485):
        tab3, tab4 = st.tabs(["Competitor Comparison", "Recommended Features & Benefits"])
        ct3 = tab3.container(height=395, border=False)
        ct3.markdown(enyk["competitor_comparison"])
        ct4 = tab4.container(height=395, border=False)
        ct4.markdown(enyk["recommended_features_benefits"])


## Setup
setup_streamlit()
# create_env_file()
pass_openAI_key()

## Main
# Display the user input fields
input_empty = st.empty()
input_container = input_empty.container()
container_height = 680
with input_container:
    customer_name, customer_title, customer_company, cp1, cp2, cp3, user_input, history, competitor = get_user_input(container_height-60)


# Display the recommendations if the button is clicked
if st.session_state.clicked:
    
    # Prepare the display
    # st.divider()
    input_empty.empty()
    st.markdown("##### Proponent Recommendations:")
    main_col1, main_col2 = st.columns([3, 5])        
    # button to go back to user input
    if st.button("Back to User Input"):
        st.session_state.clicked = False
        st.rerun()

    # Tab 2 - Enablement Center
    with main_col2:
        # Container 1: Recommendations & Feature Picker
        selected_df = st.data_editor(
            st.session_state.display_df,
            column_config=column_config_recommendations,
            column_order=["select", "customerPainPoint", "featureName", "ss_Normalized"],
            hide_index=True,
            use_container_width=True,
        )
        selected_recommendations = selected_df[selected_df["select"] == True]

        # Container 2: Sales Enablement Tools
        col2_cont = st.container(border=True, height=container_height)
        lp, salesdeck, demovideo, email = col2_cont.tabs(["Landing Page", "Sales Deck", "Demo Video", "Email"])
        # Tab 21 - Draft Email
        with email:
            st.markdown("#### Personalized Email Draft")
            email_body = generate_customized_email(selected_recommendations, user_input, customer_name, customer_title, customer_company, history, competitor)
            st.write_stream(email_body)
            # email_body = "Email Preview is not available in this demo deployment. Please download the PDF deck and video for the recommendations."
            # st.write(email_body)

        # Tab 22 - Build Sales Deck
        with salesdeck:
            col1, col2, col3 = st.columns([2.5, 2, 1.5])
            col1.markdown("#### Personalized Sales Deck")
            create_image_deck(selected_recommendations)
            with open("downloads/combined_PDF.pdf", "rb") as file:
                col3.download_button(
                    label="Download PDF Deck",
                    data=file.read(),
                    file_name="customized_deck.pdf",
                    mime="application/pdf",
                )
            
            if os.path.exists("downloads/combined_PDF.pdf"):
                displayPDF("downloads/combined_PDF.pdf", st)
            else:
                st.error("Error generating PDF. Please try again or contact me at prashant@yourproponent.com if this persists.")

        # Tab 23 - Build Demo Video
        with demovideo:
            col1, col2, col3 = st.columns([2.5, 2, 1.5])
            col1.markdown("#### Personalized Demo Video")
            # create_video(selected_recommendations) # Uncomment this line in local deployment to enable video generation
            # st.warning("Video generation is not available in demo. Below preview is pre-generated.") # Comment this line in local deployment
            if os.path.exists("downloads/video.mp4"):
                with open("downloads/video.mp4", "rb") as file:
                    col3.download_button(
                        label="Download MP4 File",
                        data=file.read(),
                        file_name="downloads/video.mp4",
                        mime="video/mp4",
                    )
                st.video("downloads/video.mp4")
            else:
                st.error("Error generating video. Please try again or contact me at prashant@yourproponent.com if this persists.")

        # Tab 24 - Generate HTML
        with lp:
            col1, col2, col3 = st.columns([2.5, 2, 1.5])
            col1.markdown("#### Personalized Landing Page")
            # Generate content for the HTML template using OpenAI
            hero_title, hero_description, feature_titles, value_propositions, webURL = (
                generate_content(
                    recommendations=selected_recommendations,
                    user_input=user_input,
                    customer_name=customer_name,
                    customer_title=customer_title,
                    customer_company=customer_company,
                    model="gpt-3.5-turbo-0125",
                )
            )
            # Generate HTML for feature sections
            features = [
                generate_feature_section(
                    feature_titles[i],
                    value_propositions[i],
                    webURL[i],
                )
                for i in range(len(feature_titles))
            ]
            # deine hero_images
            hero_images = ["https://imagedelivery.net/XawdbiDo2zcR8LA99WkwZA/9ae4b3c7-108b-4635-4d76-489b1d195700/website",
                        "https://dapulse-res.cloudinary.com/image/upload/f_auto,q_auto/remote_mondaycom_static/uploads/NaamaGros/WM-boards/Goals_strategy.png",
                        "https://assets-global.website-files.com/60058af53d79fbd8e14841ea/60181447286c0bee8d42171a_73dc280a-a211-4157-8e7c-b123b1d4ffa0_product_hero_animation_placeholder.png"]

            # Generate the HTML template
            html_template = generate_html_template(
                hero_title,
                hero_description,
                hero_images,
                features,
            )
            # Save the generated HTML template to a file
            with open("downloads/index.html", "w") as file:
                file.write(html_template)

            # Download the generated HTML template
            with open("downloads/index.html", "rb") as file:
                col3.download_button(
                    label="Download HTML File",
                    data=file.read(),
                    file_name="index.html",
                    mime="text/html",
                )
            # View the generated HTML template
            with st.container(height=520, border=False):
                with open("downloads/index.html", "r") as file:
                    html_template = file.read()
                components.html(html_template, height=4000)

    # Tab 1 - Customer Asks and Recommendations
    with main_col1:
        dislay_enyk()