import time
import streamlit as st
from assets.code.utils import check_password, set_page_config, get_themed_logo

def welcome_page():

    content = """

    # Welcome to Proponent

    Proponent is an AI-powered tool that empowers your revenue team to create personalized, product-led buying experiences for each customer, at scale and in just minutes.

    ## How Proponent Works

    1. Upload a customer interaction (video, audio, email, chat transcript) or directly ask Proponent about a customer's needs.
    2. Proponent uses advanced NLU to analyze the customer's unique pain points and generate tailored product recommendations.
    3. Review and select the most relevant recommendations from Proponent's suggestions.
    4. Automatically create personalized sales enablement content like presentations, emails, and demo videos.
    5. Share the content with your customer and have a meaningful, product-led conversation that resonates with their needs.

    ## Key Features

    - Personalized messaging grounded in a single source of truth
    - Easy-to-manage messaging framework for product marketers
    - Automatically generated sales enablement content
    - Valuable customer insights from Proponent's usage logs

    ## Getting Started

    1. Upload a customer interaction or directly ask Proponent about a customer's needs.
    2. Review and select the most relevant product recommendations.
    3. Generate personalized sales enablement content to share with your customer.

    ## Support

    If you have any questions or need assistance, please reach out to our support team at support@proponent.ai. We're here to help you succeed.

    The Proponent Team"""
    st.markdown(content)

def welcome_page_2():

    # Main title
    st.title("Welcome to Proponent")

    # Introduction
    st.write(
        "Proponent is an AI-powered tool that empowers your revenue team to create personalized, "
        "product-led buying experiences for each customer, at scale and in just minutes."
    )

    # How Proponent Works
    st.header("How Proponent Works")
    st.write(
        "1. Upload a customer interaction (video, audio, email, chat transcript) or directly ask Proponent about a customer's needs.\n"
        "2. Proponent uses advanced NLU to analyze the customer's unique pain points and generate tailored product recommendations.\n"
        "3. Review and select the most relevant recommendations from Proponent's suggestions.\n"
        "4. Automatically create personalized sales enablement content like presentations, emails, and demo videos.\n"
        "5. Share the content with your customer and have a meaningful, product-led conversation that resonates with their needs."
    )

    # Key Features
    st.header("Key Features")
    st.write(
        "- Personalized messaging grounded in a single source of truth\n"
        "- Easy-to-manage messaging framework for product marketers\n"
        "- Automatically generated sales enablement content\n"
        "- Valuable customer insights from Proponent's usage logs"
    )

    # Getting Started
    st.header("Getting Started")
    st.write(
        "1. Upload a customer interaction or directly ask Proponent about a customer's needs.\n"
        "2. Review and select the most relevant product recommendations.\n"
        "3. Generate personalized sales enablement content to share with your customer."
    )

    # Support
    st.header("Support")
    st.write(
        "If you have any questions or need assistance, please reach out to our support team at "
        "support@proponent.ai. We're here to help you succeed."
    )

    st.write("The Proponent Team")

def welcome_page_3():

    # Main title
    st.title("Welcome to Proponent")

    # Lets Get Started
    if st.button("Let's Get Started"):
        st.switch_page("pages/1_Home.py")

    # Introduction
    intro_container = st.container(border=True)
    with intro_container:
        st.write(
            "Proponent is an AI-powered tool that empowers your revenue team to create personalized, "
            "product-led buying experiences for each customer, at scale and in just minutes."
        )

    # How Proponent Works
    how_it_works_container = st.container(border=True)
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
    key_features_container = st.container(border=True)
    with key_features_container:
        st.header("Key Features")
        st.write(
            "- Personalized messaging grounded in a single source of truth\n"
            "- Easy-to-manage messaging framework for product marketers\n"
            "- Automatically generated sales enablement content\n"
            "- Valuable customer insights from Proponent's usage logs"
        )

    # Getting Started
    getting_started_container = st.container(border=True)
    with getting_started_container:
        st.header("Getting Started")
        st.write(
            "1. Upload a customer interaction or directly ask Proponent about a customer's needs.\n"
            "2. Review and select the most relevant product recommendations.\n"
            "3. Generate personalized sales enablement content to share with your customer."
        )

    # Support
    support_container = st.container(border=True)
    with support_container:
        st.header("Support")
        st.write(
            "If you have any questions or need assistance, please reach out to our support team at "
            "support@proponent.ai. We're here to help you succeed."
        )

        st.write("The Proponent Team")

# Setup
set_page_config(page_title="Proponent", layout="centered",initial_sidebar_state="collapsed")

login = st.empty()
with login:
    if not check_password():
        st.stop()
    else:
        login.empty()

get_themed_logo()
welcome_page_3()
# welcome_page()
# st.switch_page("pages/1_Home.py")


