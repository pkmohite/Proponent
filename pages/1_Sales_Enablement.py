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
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = None
    # Pass a variable to the set_page_config function
    set_page_config(page_title="Proponent", layout="wide", initial_sidebar_state="collapsed")

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
    # Initialize variables
    industry, deal_stage, deal_amount = None, None, None
    # Create a grid of customer names
    cols = st.columns(3)
    for i, customer_name in enumerate(company_db["company_name"]):
        with cols[i % 3]:
            cont = st.container(border=True, height=320)
            # Display the customer name
            cont.markdown(f"##### {customer_name}")

            # Fetch industry, deal_stage, and deal_amount for the selected customer
            industry = company_db[company_db['company_name'] == customer_name]['company_industry'].values[0]
            deal_stage = company_db[company_db['company_name'] == customer_name]['deal_stage'].values[0]
            deal_amount = company_db[company_db['company_name'] == customer_name]['deal_amount'].values[0]

            # Display contacts
            contacts_list = contacts_db[contacts_db['company_name'] == customer_name]['contact_name'].tolist()
            cont.markdown("**Contacts**")
            cont.write(", ".join(contacts_list))
            
            # Display industry, deal_stage, and deal_amount in the container=
            cont.markdown(f"**Industry:** {industry}")
            cont.markdown(f"**Deal Amount:** {deal_amount}")
            cont.markdown(f"**Deal Stage:** {deal_stage}")

            if cont.button(f"Open {customer_name}"):
                st.session_state.customer_name = customer_name
                break

    return industry, deal_stage, deal_amount

def painpoint_selector():
    header = st.container(border=True, height=870)
    
    tab0, tab1, tab2, tab3, tab4 = header.tabs(["Recommendations","AI Product Expert", "Case Studies", "Competitors", "Content Center"])

    # Display the recommendations
    tab0.markdown("##### Feature Recommendations")
    # header.info("Proponent evaluates customer interactions to create targeted recommendations. Select or remove recommendations to customize your content.", icon="ℹ️")
    cont = tab0.container(border=False, height=740)

    # Display the recommendations
    pp_data = company_pp[company_pp['company_name'] == st.session_state.customer_name]
    pp_data = pp_data.merge(mf_data, on='painPointId', how='left')
    pp_data['select'] = [True] * 5 + [False] * (len(pp_data) - 5)

    selected_recommendations = []
    
    # Title row
    cont.container(border=True)
    cols = cont.columns([1, 5, 3, 2])
    with cols[0]:
        st.write("**Select**")
    with cols[1]:
        st.write("**Customer Quote**")
    with cols[2]:
        st.write("**Feature**")
    with cols[3]:
        st.write("**Relevance**")

    # Recommendation rows
    for index, row in pp_data.iterrows():
        with cont.container(border=True):
            cols = st.columns([1, 5, 3, 2])
            
            with cols[0]:
                select = st.checkbox("Select", value=row['select'], key=f"select_{index}", label_visibility="collapsed")
            
            with cols[1]:
                st.write(f"\"{row['quote']}\"")
            
            with cols[2]:
                st.write(row['featureName'])
            
            with cols[3]:
                relevance_score = row['relevance_score']
                if relevance_score == "High":
                    st.write(f"<span style='color:#008000'>{relevance_score}</span>", unsafe_allow_html=True)
                elif relevance_score == "Medium":
                    st.write(f"<span style='color:#FFD700'>{relevance_score}</span>", unsafe_allow_html=True)
                elif relevance_score == "Low":
                    st.write(f"<span style='color:#FF0000'>{relevance_score}</span>", unsafe_allow_html=True)
                else:
                    st.write(relevance_score)

            if select:
                selected_recommendations.append(row)

    st.session_state.selected_recommendations = pd.DataFrame(selected_recommendations)

    with tab1:
        product_chatbot_v3()

    with tab4:
        sales_content_generator()

    with tab3:
        
        # Check if a competitor exists in company_db
        competitor_name = company_db[company_db['company_name'] == st.session_state.customer_name]['recc_competitor'].values[0]
        if competitor_name:
            # Get all columns from competitors2.csv where Competitor == competitor_name
            competitor_data = pd.read_csv("assets/competitors-db.csv")
            competitor_data = competitor_data[competitor_data['Competitor'] == competitor_name]
            # Display competitor name
            st.text_input("Competitor Detected!", value=competitor_name, key="competitor_name")
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

    with tab2:
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


    return None

def customer_attributes():
    cont = st.container(border=True, height=870)
    
    # Select company
    cont.markdown(f"##### {st.session_state.customer_name}")
    # Display contacts in st.multiselect: they are in contacts_db where company_name == st.session_state.customer_name
    contacts_list = contacts_db[contacts_db['company_name'] == st.session_state.customer_name]['contact_name'].tolist()
    cont.multiselect("Contacts:", contacts_list,contacts_list)
    col1, col2, col3 = cont.columns([1, 1, 1])
    # Display industry
    col1.text_input("Industry:", value=company_db[company_db['company_name'] == st.session_state.customer_name]['company_industry'].values[0])
    # Display deal_stage
    col2.text_input("Deal Stage:", value=company_db[company_db['company_name'] == st.session_state.customer_name]['deal_stage'].values[0])
    # Display deal_amount
    col3.text_input("Deal Amount:", value=company_db[company_db['company_name'] == st.session_state.customer_name]['deal_amount'].values[0])
    
    # Customer Intelligence
    internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contact_summary = None, None, None, None, None, None
    
    tab1, tab2 = cont.tabs(["Customer Intel", "Notes/Transcripts"])

    with tab1:
        # Display history_summary from company_db where company_name == st.session_state.customer_name
        history_summary = company_db[company_db['company_name'] == st.session_state.customer_name]['history_summary'].values[0]
        st.markdown("###### Deal History")
        tab1container = st.container(border=True, height=250)
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
        # Create text area to input customer notes
        st.markdown("###### Customer Notes", help="Add notes to track customer interactions, preferences, and other relevant information. Proponent will use this information to further personalize recommendations.")
        internal_notes = company_db[company_db['company_name'] == st.session_state.customer_name]['internal_notes'].values[0]
        internal_notes = st.text_area("Internal Notes", value=internal_notes, height=200, label_visibility="collapsed")

        # Get history_email and history_chat from company_db where company_name == st.session_state.customer_name
        history_email = company_db[company_db['company_name'] == st.session_state.customer_name]['history_email'].values[0]
        history_chat = company_db[company_db['company_name'] == st.session_state.customer_name]['history_chat'].values[0]
        st.markdown("###### Interaction History", help="Proponent tracks, transcribes and injests all customer interactions to create recommendations.")
        with st.expander("Email History"):
            history_email = st.text_area("Email History", value=history_email, height=250, label_visibility="collapsed")
        with st.expander("Chat History"):
            history_chat = st.text_area("Chat History", value=history_chat, height=250, label_visibility="collapsed")
        
        tab4container = st.container(border=False, height=100)
        st.button("Update Recommendations", on_click=update_recommendations)

    return internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contacts_list


    with tab1:
        product_chatbot_v3()

    with tab2:
        sales_content_generator()

def update_recommendations():
    return None

def product_chatbot_v3():
    client = OpenAI(api_key=os.getenv("USER_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Concatenate the feature names and value propositions at the line level
    features_value_prop = "\n".join([f"{feature}: {value_prop}" for feature, value_prop in zip(st.session_state.selected_recommendations["featureName"], st.session_state.selected_recommendations["valueProposition"])])
    st.markdown("##### AI Product Expert")
    col1, col2 = st.columns([6, 1])
    prompt = col1.chat_input("Ask the Product Expert")
    if col2.button("Reset Chat"):
        st.session_state.messages = []
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        messagecontainer = st.container(height=630, border=False)
        with messagecontainer.chat_message("assistant"):
            # Create a personalized prompt based on customer details
            personalized_prompt = f""" You are a product marketing expert. Use the following context to offer short helpful answers to questions that a sales person from Monday.com may ask you. 
            Response must be under 300 words. You MUST cite source:\n\n Company Name: {st.session_state.customer_name}\nCompany Industry: {industry}\n
            Deal Stage: {deal_stage}\nDeal Amount: {deal_amount}\n\nInternal Notes: {internal_notes}\nEmail History: {history_email}\nChat History: {history_chat}\n
            Competitor Battlecard: {competitor_battlecard}\nCase Studies: {case_studies_concat} \nSummary of Contact's Involved in the deal and thier needs: {contacts_list}\n 
            Recommended Features: {features_value_prop}\n\nSales Person Query: {prompt}\n\n "
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
    st.markdown("##### Content Center")
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
    # Display important links and resources
    if col4.button("Important Links & Resources"):
        display_resources()

def create_landing_page():

    def generate_lp_content(model="gpt-3.5-turbo-0125"):
        # Generate content for the HTML template using OpenAI
        hero_title, hero_description, feature_titles, value_propositions, webURL = generate_content2(
            recommendations=st.session_state.selected_recommendations,
            industry=industry,
            history_email=history_email,
            history_chat=history_chat,
            internal_notes=internal_notes,
            contact_summary=contacts_list,
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

        return html_template

    html_template = generate_lp_content()
    
    col1, col2, col3 = st.columns([2.5, 3, 1.5])
    col1.markdown("#### Personalized Landing Page")
    col3.download_button(
            label="Download HTML File",
            data=html_template.encode(),
            file_name="index.html",
            mime="text/html",
        )

    # View the generated HTML template
    with st.container(height=620, border=False):
        components.html(html_template, height=4000)

def create_sales_deck():
    col1, col2, col3 = st.columns([2.5, 3, 1.5])
    col1.markdown("#### Personalized Sales Deck")
    create_image_deck(st.session_state.selected_recommendations)
    with open("downloads/combined_PDF.pdf", "rb") as file:
        col3.download_button(
            label="Download PDF Deck",
            data=file.read(),
            file_name="customized_deck.pdf",
            mime="application/pdf",
        )
    
    if os.path.exists("downloads/combined_PDF.pdf"):
        displayPDF("downloads/combined_PDF.pdf", width=1000, height=640)
    else:
        st.error("Error generating PDF. Please try again or contact me at prashant@proponentapp.com if this persists.")

def create_demo_video():
    # create_video(st.session_state.selected_recommendations) # Uncomment this line in local deployment to enable video generation
    st.warning("This feature is disabled in the demo version. Please contact me at prashant@proponentapp.com to see a demo.")
    col1, col2, col3 = st.columns([2.5, 3, 1.5])
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
        st.error("Error generating video. Please try again or contact me at prashant@proponentapp.com if this persists.")

def display_resources():
    # Fetch files-db.csv
    files_db = pd.read_csv("assets/files-db.csv")
    
    # Join files_db with selected_recommendations on painPointId
    resources = files_db.merge(st.session_state.selected_recommendations, on="painPointId", how="inner")
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


# Select Deal
deal_selector = st.empty()
deal_container = deal_selector.container()
with deal_container:
    st.write("### Active Deals")
    industry, deal_stage, deal_amount = customer_selector()
# Check if customer_name is selected
if st.session_state.customer_name is None:
    st.stop()

## Main
deal_selector.empty()
mainco1, mainco2 = st.columns([4,6])

# Display customer details
with mainco1:
    internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contacts_list = customer_attributes()

# Display recommendations
with mainco2:
    # Table to display and select recommendations
    painpoint_selector()

    # Content center: sales chatbot and content generator
    # content_center()


