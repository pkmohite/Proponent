import streamlit as st
import os
import json
import openai
from dotenv import load_dotenv

def pass_openAI_key():
    if "USER_API_KEY" in os.environ:
        openai.api_key = os.getenv("USER_API_KEY")
    else:
        st.error("OpenAI API key not found. Please set the API key in the Setting page.")


def set_API_key():
    
    # Add your code for the LLM settings tab here
    st.subheader("Set API Key")
    con = st.empty()
    st.session_state.api_key = con.text_input("Enter your API key", value=os.getenv("USER_API_KEY"), type="password", key = '1')
    b1,b2,b3 = st.columns([.5,1,8])

    # Save API key to environment variable
    if b1.button("Save"):
        pass_openAI_key()
        os.environ["USER_API_KEY"] = st.session_state.api_key
        st.write("API key saved!")
    if b2.button("Save key in .env"):
        pass_openAI_key()
        os.environ["USER_API_KEY"] = st.session_state.api_key
        with open(".env", "r+") as file:
            lines = file.readlines()
            file.seek(0)
            key_exists = False
            for line in lines:
                if line.startswith("USER_API_KEY="):
                    file.write(f"USER_API_KEY={st.session_state.api_key}\n")
                    key_exists = True
                else:
                    file.write(line)
            if not key_exists:
                file.write(f"USER_API_KEY={st.session_state.api_key}\n")
            file.truncate()
        st.write("API key saved in .env file!")

    if b3.button("Delete key in .env"):
        os.environ["USER_API_KEY"] = ""
        with open(".env", "r+") as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                if not line.startswith("USER_API_KEY="):
                    file.write(line)
            file.truncate()
        st.write("API key deleted from .env file!")
        con.text_input("Enter your API key", value=os.getenv("USER_API_KEY"), type="password", key = '2')   


# Create tabs
tab1, tab2 = st.tabs(["General", "LLM"])
with tab1:
    st.subheader("General Settings")
    # Add your code for the general settings tab here
with tab2:
    set_API_key()


  