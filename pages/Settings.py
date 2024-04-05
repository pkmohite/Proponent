import streamlit as st
import os
import json
import openai
from dotenv import load_dotenv
import pandas as pd

# Function to pass the OpenAI key
def pass_openAI_key():
    if "USER_API_KEY" in os.environ:
        openai.api_key = os.getenv("USER_API_KEY")
    else:
        st.error("OpenAI API key not found. Please set the API key in the Setting page.")

# Tab 2: LLM settings
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


# Tab 3: Customer Personas
def update_customer_personas():
    st.write("Upload a CSV file with columns category name, persona name, and persona description.")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Preview the uploaded file using st.write
        df = pd.read_csv(uploaded_file)
        st.data_editor(df, hide_index=True)
        # Add a button to save the uploaded file as a json file
        if st.button("Save as JSON"):
            df_json = df.groupby("category name").apply(lambda x: x[["persona name", "persona description"]].to_dict(orient="records")).to_dict()
            with open("assets/customer_personas.json", "w") as file:
                json.dump(df_json, file)
                st.write("File saved as JSON!")


def read_themes_csv():
    themes_df = pd.read_csv("assets/themes.csv")
    theme_names = themes_df["themeName"].tolist()
    selected_theme = st.selectbox("Select a theme", theme_names)
    selected_theme_values = themes_df.loc[themes_df["themeName"] == selected_theme].iloc[0]
    # apply button that updates the selected theme values in streamlit/config.toml
    if st.button("Apply"):
        st.write(f"Theme {selected_theme} applied!")
        # Add the code to update the theme in the config.toml file here
        with open(".streamlit/config.toml", "w") as file:
            file.write("[theme]\n")
            file.write(f'primaryColor="{selected_theme_values["primaryColor"]}"\n')
            file.write(f'backgroundColor="{selected_theme_values["backgroundColor"]}"\n')
            file.write(f'secondaryBackgroundColor="{selected_theme_values["secondaryBackgroundColor"]}"\n')
            file.write(f'textColor="{selected_theme_values["textColor"]}"\n')
            file.write(f'font="{selected_theme_values["font"]}"\n')
        st.rerun()



# Create tabs
tab1, tab2, tab3 = st.tabs(["General", "LLM", "Customer Personas"])
with tab1:
    read_themes_csv()
with tab2:
    set_API_key()

with tab3:
    st.subheader("Customer Personas")
    update_customer_personas()


  