import streamlit as st
import os
import json
import pandas as pd
from assets.code.utils import pass_openAI_key

# Tab 2: LLM settings
def set_API_key():
    
    # Add your code for the LLM settings tab here
    st.subheader("Set API Key")
    con = st.empty()
    st.session_state.api_key = con.text_input("Enter your API key", value=os.getenv("USER_API_KEY"), type="password", key = '1')
    b1,b2,b3 = st.columns([1,1,6])

    # Save API key to environment variable
    if b1.button("Save API Key"):
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

    if b2.button("Delete API Key"):
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
    st.subheader("Customer Personas")
    st.write("Upload a CSV file with columns category name, persona name, and persona description.")
    # Add a download button for template
    col1, col2 = st.columns([10, 1])
    template_csv = "assets/templates/mf_template.csv"
    col2.download_button("Download CSV Template", template_csv, file_name="mf_template.csv")
    uploaded_file = col1.file_uploader("Upload CSV File:", label_visibility= 'collapsed', type=["csv"])
    
    if uploaded_file is not None:
        # Preview the uploaded file using st.write
        df = pd.read_csv(uploaded_file)
        edited_data = st.data_editor(df, hide_index=True)
        
        # Add a button to save the uploaded file as a json file
        if st.button("Save as JSON"):
            df_json = edited_data.groupby("category_name").apply(lambda x: x[["persona_name", "persona_description"]].to_dict(orient="records")).to_dict()
            with open("assets/customer_personas.json", "w") as file:
                json.dump(df_json, file)
                st.write("File saved as JSON!")


def update_themes_csv():
    themes_df = pd.read_csv("assets/themes.csv")
    theme_names = themes_df["themeName"].tolist()
    # get the index of the row where active = 'x'
    current_theme_index = int(themes_df[themes_df["active"] == "x"].index[0])
    st.markdown("### Set Theme")
    selected_theme = st.selectbox("Change Proponent Theme", theme_names,label_visibility="collapsed", index= current_theme_index)
    selected_theme_values = themes_df.loc[themes_df["themeName"] == selected_theme].iloc[0]
    # apply button that updates the selected theme values in streamlit/config.toml
    if st.button("Apply"):
        # Add the code to update the theme in the config.toml file here
        with open(".streamlit/config.toml", "w") as file:
            file.write("[theme]\n")
            file.write(f'primaryColor="{selected_theme_values["primaryColor"]}"\n')
            file.write(f'backgroundColor="{selected_theme_values["backgroundColor"]}"\n')
            file.write(f'secondaryBackgroundColor="{selected_theme_values["secondaryBackgroundColor"]}"\n')
            file.write(f'textColor="{selected_theme_values["textColor"]}"\n')
            file.write(f'font="{selected_theme_values["font"]}"\n')        
        st.write(f"Theme {selected_theme} applied!")
        # update the active column in the themes.csv file
        themes_df["active"] = ""
        themes_df.loc[themes_df["themeName"] == selected_theme, "active"] = "x"
        themes_df.to_csv("assets/themes.csv", index=False)
        st.rerun()


# Setuo
st.set_page_config(page_title="Settings", page_icon=":gear:", layout="wide")
st.title("Settings")

# Create tabs
tab1, tab2, tab3 = st.tabs(["General", "LLM", "Customer Personas"])
with tab1:
    update_themes_csv()
with tab2:
    set_API_key()

with tab3:
    update_customer_personas()


  