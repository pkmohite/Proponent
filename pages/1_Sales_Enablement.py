import os
import pandas as pd
import streamlit as st
from assets.code.element_configs import config_painpoint_selector
from assets.code.utils import pass_openAI_key, create_image_deck, displayPDF, verify_password, set_page_config
from moviepy.editor import VideoFileClip, concatenate_videoclips
from assets.code.genHTML import generate_content2, generate_feature_section, generate_html_template
import streamlit.components.v1 as components
from openai import OpenAI


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
        st.markdown("###### Deal History")
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
        display_resources()

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
    create_video(selected_recommendations) # Uncomment this line in local deployment to enable video generation
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

def display_resources():
    # Fetch files-db.csv
    files_db = pd.read_csv("assets/files-db.csv")
    
    # Join files_db with selected_recommendations on painPointId
    resources = files_db.merge(selected_recommendations, on="painPointId", how="inner")
    resources = resources.sort_values(by="similarity_score", ascending=False)
    
    # Display the resources
    st.markdown("#### Links & Resources")
    resourcescont = st.container(border=False, height=560)
    col1, col2= resourcescont.columns([1,1])
    feature_names = resources['featureName'].unique()
    for i, feature_name in enumerate(feature_names):
        if i % 2 == 0:
            col = col1
        else:
            col = col2
        expander = col.expander(feature_name, expanded=True)
        feature_resources = resources[resources['featureName'] == feature_name]
        for index, row in feature_resources.iterrows():
            expander.page_link(row['hyperlink'], label=row['title'], icon=row['icon'], use_container_width=True)

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


