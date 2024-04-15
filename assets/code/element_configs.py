import streamlit as st
import pyarrow as pa

# Home - Recommendations Table

column_config_recommendations = {
    "select": st.column_config.Column(label="Select", disabled=False, width="small"),
    "featureName": st.column_config.Column(label="Feature Name", disabled=True, width="medium"),
    "valueProposition": st.column_config.Column(
        label="Value Proposition", disabled=True, width="medium"
    ),
    "ss_Normalized": st.column_config.ProgressColumn(label="Similarity Score"),
    "customerPainPoint": None,
    "similarity_score": None,
    "embedding": None,
    "painPointId": None,
    "pdfFile": None,
    "videoFile": None,
    "webURL": None,
    "PDF_Present": None,
    "Video_Present": None,
    "Web_URL_Present": None,
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
        "webURL": st.column_config.Column(label="Web URL", width="medium"),
    }


column_config_edit = {
        "painPointId": st.column_config.Column(label="ID", width="small"),
        "embedding": st.column_config.Column(label="Embedding", width="small"),
        "pdfFile": st.column_config.Column(label="PDF File", width="small"),
        "videoFile": st.column_config.Column(label="Video File", width="small"),
        "webURL": st.column_config.Column(label="Web URL", width="small"),
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

analytics_column_config = {
    "painPointId": None,
    "customerPainPoint": st.column_config.Column(
        label="Customer Pain Point", width="medium"
    ), 
    "featureName": st.column_config.Column(
        label="Feature Name", width="medium"
    ),
    "score": st.column_config.Column(label="Score", width="small"),
    "count": st.column_config.Column(label="Count", width="small"),
    "percentage": st.column_config.Column(label="Percentage", width="small"),
}

default_theme = {
    "primaryColor": "#F63366",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#262730",
    "font": "sans serif"
}

parquet_schema_log = pa.schema([
    pa.field("customer_name", pa.string()),
    pa.field("customer_title", pa.string()),
    pa.field("customer_company", pa.string()),
    pa.field("persona_category1", pa.string()),
    pa.field("persona_category2", pa.string()),
    pa.field("persona_category3", pa.string()),
    pa.field("user_input", pa.string()),
    pa.field("paintpoints", pa.list_(pa.int64())),
    pa.field("date", pa.string()),
    pa.field("time", pa.string())
    ])

parquet_schema_mf = pa.schema([
    pa.field("painPointId", pa.int64()),
    pa.field("customerPainPoint", pa.string()),
    pa.field("featureName", pa.string()),
    pa.field("valueProposition", pa.string()),
    pa.field("pdfFile", pa.string(), nullable=True),
    pa.field("videoFile", pa.string(), nullable=True),
    pa.field("webURL", pa.string(), nullable=True),
    pa.field("embedding", pa.list_(pa.float64()), nullable=True)
])