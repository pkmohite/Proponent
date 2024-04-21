import time
import streamlit as st
from assets.code.utils import check_password, set_page_config, get_themed_logo
import streamlit as st

def intro_page():
    
    ## Introduction
    col1,col2 = st.columns([1,1])
    with col1:
        st.container(height=100,border=False)
        st.header("Welcome to Proponent!")
        st.write("Read this guide to learn how Proponent can help you create personalized, product-led buying experiences for your customers.")
        if st.button("Done reading? Click here to get started"):
            st.switch_page("pages/1_Home.py")
    
    with col2:
        st.image("assets/images/kelly-kapoor.svg", use_column_width=True)
    
    ## Step by Step Guide
    st.container(height=30,border=False)
    st.markdown("### Quick Start Guide")
    with st.expander("Step 1: Upload a Customer Interaction"):
        st.write("Upload a video, audio, email, or chat transcript to Proponent. Alternatively, you can directly ask Proponent about a customer's needs.")
        st.image("assets/images/kelly-kapoor.svg", use_column_width=True)

    with st.expander("Step 2: Select a Customer to Personalize Recommendations and click 'Get Recommendations'"):
        st.write("Proponent uses advanced NLU to analyze the customer's unique pain points and generate tailored product recommendations. Review and select the most relevant recommendations from Proponent's suggestions.")
        st.image("assets/images/kelly-kapoor.svg", use_column_width=True)

    with st.expander("Step 3: Review and shortlist recommendations that you want to include in the sales enablement content"):
        st.write("Proponent uses advanced NLU to analyze the customer's unique pain points and generate tailored product recommendations. Review and select the most relevant recommendations from Proponent's suggestions.")
        st.image("assets/images/kelly-kapoor.svg", use_column_width=True)

    with st.expander("Step 4: Click 'Generate Content' to create personalized sales enablement content for the customer!"):
        st.write("Automatically create personalized sales enablement content like presentations, emails, and demo videos. Share the content with your customer and have a meaningful, product-led conversation that resonates with their needs.")
        st.image("assets/images/kelly-kapoor.svg", use_column_width=True)

    # Additional Steps: Analytics
    with st.expander("Step 5: Get Insights into Customer Pain Points and Preferences in the Analytics Dashboard"):
        st.write("Gain valuable insights into customer pain points and preferences by analyzing Proponent's usage logs. Use these insights to refine your messaging and product recommendations.")
        st.image("assets/images/kelly-kapoor.svg", use_column_width=True)

    ## What Can You Do with Proponent?
    st.container(height=30,border=False)
    st.markdown("### What Can You Do with Proponent?")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True, height=200):
            st.markdown("##### Product Recommendations")
            st.write("Generate highly relevant product recommendations based on customer interactions and preferences.")

    with col2:
        with st.container(border=True, height=200):
            st.markdown("##### Sales Deck")
            st.write("Create a customized sales deck tailored to each customer's specific needs and pain points.")

    with col3:
        with st.container(border=True, height=200):
            st.markdown("##### Landing Page")
            st.write("Automatically generate a personalized landing page highlighting the most relevant features and benefits.")

    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True, height=200):
            st.markdown("##### Product Demo Video")
            st.write("Showcase your product's capabilities with an interactive demo that focuses on the customer's key interests.")

    with col5:
        with st.container(border=True, height=200):
            st.markdown("##### Email & Chat Templates")
            st.write("Craft compelling email templates that effectively communicate the value proposition to the customer.")

    with col6:
        with st.container(border=True, height=200):
            st.markdown("##### Customer Persona Insights")
            st.write("Gain insights into the effectiveness of your personalized approach with a comprehensive analytics dashboard.")

    # Contact Us
    st.container(height=30,border=False)
    st.markdown("### Contact Us")
    st.write("For any questions or assistance, please reach out through the following channels:")
    col1, col2, col3, col4 = st.columns([.8,.9,1,5])
    col1.link_button(":email: Email", "mailto:prashant@yourproponent.com")
    col2.link_button(":globe_with_meridians: Website", "https://yourproponent.com")
    col3.link_button(":link: LinkedIn", "https://www.linkedin.com/company/proponent-ai")
    


def welcome_page():

    # Main title
    st.title("Welcome to Proponent")

    # Lets Get Started
    if st.button("Let's Get Started"):
        st.switch_page("pages/1_Home.py")

    # Introduction
    intro_container = st.container(border=True, height=200)
    with intro_container:
        st.write(
            "Proponent is an AI-powered tool that empowers your revenue team to create personalized, "
            "product-led buying experiences for each customer, at scale and in just minutes."
        )

    # How Proponent Works
    how_it_works_container = st.container(border=True, height=200)
    with how_it_works_container:
        st.header("How Proponent Works")
        st.write(
            "1. Upload a customer interaction (video, audio, email, chat transcript) or directly ask Proponent about a customer's needs.\n"
            "2. Proponent uses advanced NLU to analyze the customer's unique pain points and generate tailored product recommendations.\n"
            "3. Review and select the most relevant recommendations from Proponent's suggestions.\n"
            "4. Automatically create personalized sales enablement content like presentations, emails, and demo videos.\n"
            "5. Share the content with your customer and have a meaningful, product-led conversation that resonates with their needs."
        )

    # Key Features
    key_features_container = st.container(border=True, height=200)
    with key_features_container:
        st.header("Key Features")
        st.write(
            "- Personalized messaging grounded in a single source of truth\n"
            "- Easy-to-manage messaging framework for product marketers\n"
            "- Automatically generated sales enablement content\n"
            "- Valuable customer insights from Proponent's usage logs"
        )

    # Getting Started
    getting_started_container = st.container(border=True, height=200)
    with getting_started_container:
        st.header("Getting Started")
        st.write(
            "1. Upload a customer interaction or directly ask Proponent about a customer's needs.\n"
            "2. Review and select the most relevant product recommendations.\n"
            "3. Generate personalized sales enablement content to share with your customer."
        )

    # Support
    support_container = st.container(border=True, height=200)
    with support_container:
        st.markdown("##### Support")
        st.write(
            "If you have any questions or need assistance, please reach out to our support team at "
            "support@proponent.ai. We're here to help you succeed."
        )

        st.write("The Proponent Team")

# Setup
set_page_config(page_title="Proponent", layout="wide",initial_sidebar_state="collapsed")

# Check password
col1,col2,col3 = st.columns([1,1.5,1])
with col2:
    col2a, col2b, col2c = st.columns([1,3,1])
    if not check_password():
        st.stop()

# # Logo
# log1,log2,log3 = st.columns([3,1,3])
# with log2:
#     get_themed_logo()

# Welcome Page
col1,col2,col3 = st.columns([1,3,1])
with col2:
    # welcome_page()
    # Run the intro page
    intro_page()


