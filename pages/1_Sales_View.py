import os, hmac, json, time, ast
import pandas as pd
import streamlit as st
from assets.code.element_configs import column_config_recommendations, config_about, config_painpoint_selector
from assets.code.utils import generate_customized_email, pass_openAI_key, get_embedding, calculate_similarity_ordered, transcribe_video, load_mf_data
from assets.code.utils import create_summary, update_log_parquet, create_image_deck, displayPDF, verify_password, set_page_config, generate_enyk
from moviepy.editor import VideoFileClip, concatenate_videoclips
from assets.code.genHTML import generate_content2, generate_feature_section, generate_html_template
import streamlit.components.v1 as components
from openai import OpenAI

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
    
    # Store recommendations and AI generated content in session state
    # st.session_state.summary = summary
    st.session_state.display_df = df_formatted.head(7)
    st.session_state.enyk = generate_enyk(df_formatted.head(7), user_input, customer_name, customer_title, customer_company, history, competitor)
    st.session_state.email_body = generate_customized_email(df_formatted.head(5), user_input, customer_name, customer_title, customer_company, history, competitor)
    generate_lp_content(df_formatted.head(5), user_input, customer_name, customer_title, customer_company)
    
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


#################### New Code ####################

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
    set_page_config(page_title="Proponent", layout="wide", initial_sidebar_state="expanded")
    # Verify the password
    verify_password()
    # Set the page logo
    # get_themed_logo()

def load_data():
    # Load company-db.csv file
    company_db = pd.read_csv("assets/company-db.csv")

    # Load company-pp.csv file
    company_pp = pd.read_csv("assets/company-pp.csv")

    # Load contacts-db.csv file
    contacts_db = pd.read_csv("assets/contacts-db.csv")

    # Load mf_embeddings.parquet
    mf_data = pd.read_parquet("assets/mf_embeddings.parquet")

    return company_db, company_pp, contacts_db, mf_data

def customer_selector():
    cont = st.container(border=True)
    # Initialize variables
    industry, deal_stage, deal_amount = None, None, None

    # Select company
    cont.markdown("##### Step1: Select a Customer Account")
    cont.info("Proponent syncs with your CRM to fetch customer data.",icon="ℹ️")
    st.session_state.customer_name = cont.selectbox("Select Customer", company_db["company_name"].tolist(), index=None, placeholder="Select Customer Name")
    
    if st.session_state.customer_name:
        # Display contacts in st.multiselect: they are in contacts_db where company_name == st.session_state.customer_name
        contacts_list = contacts_db[contacts_db['company_name'] == st.session_state.customer_name]['contact_name'].tolist()
        contacts = cont.multiselect("Contacts:", contacts_list,contacts_list)
        col1, col2, col3 = cont.columns([1, 1, 1])
        # Display industry
        industry = col1.text_input("Industry:", value=company_db[company_db['company_name'] == st.session_state.customer_name]['company_industry'].values[0])
        # Display deal_stage
        deal_stage = col2.text_input("Deal Stage:", value=company_db[company_db['company_name'] == st.session_state.customer_name]['deal_stage'].values[0])
        # Display deal_amount
        deal_amount = col3.text_input("Deal Amount:", value=company_db[company_db['company_name'] == st.session_state.customer_name]['deal_amount'].values[0])

    return industry, deal_stage, deal_amount

def painpoint_selector():
    cont = st.container(border=True)
    # Display the recommendations
    cont.markdown("##### Step 2: View & Select Recommendations")
    cont.info("Proponent evaluates customer interactions to create targeted recommendations. Select or remove recommendations to customize your content.", icon="ℹ️")
    
    
    # Display the recommendations
    pp_data = company_pp[company_pp['company_name'] == st.session_state.customer_name]
    pp_data = pp_data.merge(mf_data, on='painPointId', how='left')
    pp_data['select'] = [True] * 5 + [False] * (len(pp_data) - 5)

    selected_df = cont.data_editor(
            pp_data,
            column_config=config_painpoint_selector,
            column_order=["select", "customerPainPoint", "featureName", "similarity_score"],
            hide_index=True,
            use_container_width=True,
        )
    
    selected_recommendations = selected_df[selected_df["select"] == True]

    return selected_recommendations

def customer_attributes():
    cont = st.container(border=True, height=1000)
    # Initialize variables
    internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contact_summary = None, None, None, None, None, None
    
    cont.markdown("##### Step 3: Tactical Deep Dive ")
    cont.info("Proponent tracks customer interactions to surface high impact sales content and information like competitor battlecards, historical needs, and customer success stories in similar domains", icon="ℹ️")
    
    tab1, tab2, tab3, tab4 = cont.tabs(["Customer Intel", "Competitor Intelligence", "Case Studies", "Notes/Transcripts"])

    with tab1:
        # Display history_summary from company_db where company_name == st.session_state.customer_name
        history_summary = company_db[company_db['company_name'] == st.session_state.customer_name]['history_summary'].values[0]
        st.markdown("###### ?")
        tab1container = st.container(border=True, height=300)
        tab1container.markdown(history_summary)
        
        st.markdown("###### What are the needs of the contacts involved in the deal?")
        # Display all contact_name,company_name,title,history_summary from contacts_db where company_name == st.session_state.customer_name
        for index, row in contacts_db[contacts_db['company_name'] == st.session_state.customer_name].iterrows():
            contact_name = row['contact_name']
            title = row['title']
            contact_summary = row['history_summary']
            buyer_persona = row['buyer_persona']
            with st.expander(f"{contact_name}, {title}"):
                st.markdown(f"**Buyer Persona:** {buyer_persona}")
                st.write(contact_summary)
        # Create a mega string by concatenating all contact summaries
        contact_summary = "\n".join([f"{contact_name}, {title}: {contact_summary}" for contact_name, title, contact_summary in zip(contacts_db['contact_name'], contacts_db['title'], contacts_db['history_summary'])])

    with tab2:
        
        # Check if a competitor exists in company_db
        competitor_name = company_db[company_db['company_name'] == st.session_state.customer_name]['recc_competitor'].values[0]
        if competitor_name:
            # Get all columns from competitors2.csv where Competitor == competitor_name
            competitor_data = pd.read_csv("assets/competitors-db.csv")
            competitor_data = competitor_data[competitor_data['Competitor'] == competitor_name]
            # Display competitor name
            st.text_input("Competitor Detected!", value=competitor_name)
            col1,col2,col3 = st.columns([1,5,1])
            col2.image(competitor_data['Image URL'].values[0], use_column_width=True)
            with st.expander("Strengths"):
                st.write(competitor_data['Key Strengths'].values[0])
            with st.expander("Weaknesses"):
                st.write(competitor_data['Key Weaknesses'].values[0])
            with st.expander("Unique Selling Points"):
                st.write(competitor_data['Unique Selling Points'].values[0])
            with st.expander("Pricing"):
                st.write(competitor_data['Pricing'].values[0])
            
            # Concatenate all fields into a single string
            competitor_battlecard = "\n".join([f"{field}: {competitor_data[field].values[0]}" for field in competitor_data.columns])
        else:
            st.text_input("Competitor", value="No Competitor Detected")
            competitor_name = "No Competitor Present"
    
    with tab3:
        # Get all columns from compnaies-pp.csv where company_name == st.session_state.customer_name
        case_studies = company_pp[company_pp['company_name'] == st.session_state.customer_name]
        # Merge case_studies and mf_data where painPointId == company_pp['painPointId']
        case_studies = case_studies.merge(mf_data, on='painPointId', how='left')
        # Get all unique customerName from case_studies
        customer_names = case_studies['customerName'].unique()
        # Remove blank customer names
        customer_names = [name for name in customer_names if name]

        # Load customer.csv 
        customer_db = pd.read_csv("assets/customers-db.csv")
        case_studies_concat = []
        # For each customer_name in customer_names, get data from customers.csv where customerName == customer_name
        for customer_name in customer_names:
            customer_data = customer_db[customer_db['customerName'] == customer_name]
            industry = customer_data['industry'].values[0]
            business_impact = customer_data['businessImpact'].values[0]
            customer_info = f"{customer_name} - {industry}: {business_impact}"
            case_studies_concat.append(customer_info)
            # Display customer_name, customer_logo, customer_description, customer_case_study
            with st.expander(f"{customer_name} - {industry}"):
                st.markdown("#")
                col1, col2, col3 = st.columns([1, 5, 1])
                col2.image(customer_data['imageLink'].values[0])
                st.write(business_impact)

    with tab4:
        # Create text area to input customer notes
        st.markdown("###### Customer Notes", help="Add notes to track customer interactions, preferences, and other relevant information. Proponent will use this information to further personalize recommendations.")
        internal_notes = company_db[company_db['company_name'] == st.session_state.customer_name]['internal_notes'].values[0]
        internal_notes = st.text_area("Internal Notes", value=internal_notes, height=200, label_visibility="collapsed")

        # Get history_email and history_chat from company_db where company_name == st.session_state.customer_name
        history_email = company_db[company_db['company_name'] == st.session_state.customer_name]['history_email'].values[0]
        history_chat = company_db[company_db['company_name'] == st.session_state.customer_name]['history_chat'].values[0]
        st.markdown("###### Interaction History", help="Proponent tracks, transcribes and injests all customer interactions to create recommendations.")
        tab4container = st.container(border=False, height=400)
        with tab4container.expander("Email History"):
            history_email = st.text_area("Email History", value=history_email, height=250, label_visibility="collapsed")
        with tab4container.expander("Chat History"):
            history_chat = st.text_area("Chat History", value=history_chat, height=250, label_visibility="collapsed")
        
        st.button("Update Recommendations", on_click=update_recommendations)

    return internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contact_summary

def content_center():
    cont = st.container(border=True, height=925)
    cont.markdown("##### Step 4: Content Center")
    cont.info("Proponent generates sales content and marketing collateral to help you close deals faster. Use the AI Product Expert to create a sales chatbot or the Content Generator to create a personalized content deck.", icon="ℹ️")
    tab1, tab2 = cont.tabs(["AI Product Expert", "Content Generator"])

    with tab1:
        product_chatbot_v3()

    with tab2:
        sales_content_generator()

def update_recommendations():
    return None

def product_chatbot_v3():
    client = OpenAI(api_key=os.getenv("USER_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Concatenate the feature names and value propositions at the line level
    features_value_prop = "\n".join([f"{feature}: {value_prop}" for feature, value_prop in zip(selected_recommendations["featureName"], selected_recommendations["valueProposition"])])

    col1, col2 = st.columns([6, 1])
    prompt = col1.chat_input("Say something")
    if col2.button("Reset Chat"):
        st.session_state.messages = []
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        messagecontainer = st.container(height=550, border=False)
        with messagecontainer.chat_message("assistant"):
            # Create a personalized prompt based on customer details
            personalized_prompt = f"""As a product marketing expert, use the following customer details to provide a highly targeted response:\n\n Company Name: {st.session_state.customer_name}\nCompany Industry: {industry}\n
            Deal Stage: {deal_stage}\nDeal Amount: {deal_amount}\n\nInternal Notes: {internal_notes}\nEmail History: {history_email}\nChat History: {history_chat}\n
            Competitor Battlecard: {competitor_battlecard}\nCase Studies: {case_studies_concat} \nSummary of Contact's Involved in the deal and thier needs: {contact_summary}\n Recommended Features: {features_value_prop}\n\n
            Focus on addressing the customer's specific needs, pain points, and challenges based on their industry, deal stage, and history. Highlight relevant features, 
            benefits, and case studies that demonstrate how our product can provide value and solve their problems. Tailor your language and tone to match the customer's 
            background and preferences.\n\nUser Query: {prompt}\n\nAssistant Response:"
            """
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": personalized_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

def sales_content_generator():

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    # Create a personalized landing page
    if col1.button("Create Personalized Landing Page"):
        create_landing_page()
    # Create a personalized sales deck
    if col2.button("Create Personalized Sales Deck"):
        create_sales_deck()
    # Create a demo video
    if col3.button("Create Personalized Product Demo"):
        create_demo_video()
    if col4.button("Important Links & Resources"):
        # display_resources()

def create_landing_page():

    def generate_lp_content(model="gpt-3.5-turbo-0125"):
        # Generate content for the HTML template using OpenAI
        hero_title, hero_description, feature_titles, value_propositions, webURL = generate_content2(
            recommendations=selected_recommendations,
            industry=industry,
            history_email=history_email,
            history_chat=history_chat,
            internal_notes=internal_notes,
            contact_summary=contact_summary,
            model=model,
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

    generate_lp_content()
    
    col1, col2, col3 = st.columns([2.5, 2, 1.5])
    col1.markdown("#### Personalized Landing Page")

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

def create_sales_deck():
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
        displayPDF("downloads/combined_PDF.pdf", width=810, height=560)
    else:
        st.error("Error generating PDF. Please try again or contact me at prashant@yourproponent.com if this persists.")

def create_demo_video():
    col1, col2, col3 = st.columns([2.5, 2, 1.5])
    col1.markdown("#### Personalized Demo Video")
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

## Setup
setup_streamlit()
# create_env_file()
pass_openAI_key()
# Load data
company_db, company_pp, contacts_db, mf_data = load_data()

## Main
mainco1, mainco2 = st.columns([4,6])


with mainco1:
    # Select customer/company
    industry, deal_stage, deal_amount = customer_selector()
    # Check if customer_name is selected
    if st.session_state.customer_name is None:
        st.stop()
    # Display customer details
    internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contact_summary = customer_attributes()

with mainco2:
    # Table to display and select recommendations
    selected_recommendations = painpoint_selector()

    # Content center: sales chatbot and content generator
    content_center()


