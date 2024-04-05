import streamlit as st

# Home - Recommendations Table
column_config_recommendations = {
    "Select": st.column_config.Column(label="Select", disabled=False),
    "Customer Pain Point": st.column_config.Column(
        label="Customer Pain Point", disabled=True, width="medium"
    ),
    "Feature Name": st.column_config.Column(
        label="Feature Name", disabled=True, width="medium"
    ),
    "Value Proposition": st.column_config.Column(
        label="Value Proposition", disabled=True, width="large"
    ),
    "Similarity Score": st.column_config.ProgressColumn(label="Similarity Score"),
    "PDF File": st.column_config.Column(label="PDF File", disabled=True),
    "Video File": st.column_config.Column(label="Video File", disabled=True),
    "PDF File Name": None,
    "Video File Name": None,
}

# Home - About
config_about = """
## About Proponent

Proponent is an AI-powered customer intelligence tool that creates hyper-personalized selling experiences for each customer that are centered around their unique customer persona, pain points, and business needs.

"""

# Upload messages via CSV
config_csv_upload = {
        "painPointId": None,
        "customerPainPoint": st.column_config.Column(
            label="Customer Pain Point", width="medium"
        ),
        "featureName": st.column_config.Column(
            label="Feature Name", width="medium"
        ),
        "valueProposition": st.column_config.Column(
            label="Value Proposition", width="medium"
        ),
        "pdfFile": st.column_config.Column(label="PDF File", width="medium"),
        "videoFile": st.column_config.Column(label="Video File", width="medium"),
    }


column_config_edit = {
        "painPointId": None,
        "embedding": None,
        "pdfFile": None,
        "videoFile": None,
        "selected": st.column_config.Column(label="Select", width="small"),
        "customerPainPoint": st.column_config.Column(
            label="Customer Pain Point", width="medium"
        ),
        "featureName": st.column_config.Column(
            label="Feature Name", width="medium"
        ),
        "valueProposition": st.column_config.Column(
            label="Value Proposition", width="medium"
        ),
    }


default_theme = {
    "primaryColor": "#F63366",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#262730",
    "font": "sans serif"
}
