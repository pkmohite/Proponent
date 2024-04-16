import pandas as pd
import streamlit as st
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from assets.code.element_configs import parquet_schema_log, analytics_column_config
import matplotlib.pyplot as plt
import textwrap
import altair as alt
import json
from assets.code.utils import get_mf_and_log


# Tab 1: Analytics - Create Filter Components
def create_filter_components(df,container = st):

    title = container.selectbox("Customer Title", df['customer_title'].unique(), index=None)
    a1, a2 = container.columns(2)
    from_date = a1.date_input("From Date", None)
    to_date = a2.date_input("To Date", None)
    date_range = [from_date, to_date]
       

    persona_category1 = container.selectbox("Persona Category 1", df['persona_category1'].unique(), index=None)
    persona_category2 = container.selectbox("Persona Category 2", df['persona_category2'].unique(), index=None)
    persona_category3 = container.selectbox("Persona Category 3", df['persona_category3'].unique(), index=None)

    competitors = container.selectbox("Competitors", ["Competitor 1", "Competitor 2", "Competitor 3"], index=None)
    return title, date_range, persona_category1, persona_category2, persona_category3

# Tab 1: Analytics - Get Painpoint Metrics
def get_painpoint_metrics(df, mf_content, title = None, date_range = None, persona_category1 = None, persona_category2 = None, persona_category3 = None, get_all = False):
    painpoint_cols = ['pp0', 'pp1', 'pp2', 'pp3', 'pp4', 'pp5', 'pp6']
    
    if get_all == False:
        customer_df = df[
            (df['customer_title'] == title if title else True) &
            (df['date'] >= date_range[0] if date_range[0] else True) &
            (df['date'] <= date_range[1] if date_range[1] else True) &
            (df['persona_category1'] == persona_category1 if persona_category1 else True) &
            (df['persona_category2'] == persona_category2 if persona_category2 else True) &
            (df['persona_category3'] == persona_category3 if persona_category3 else True)
        ]
    else:
        customer_df = df

    pref_scores = {}
    pref_counts = {}
    
    for i, col in enumerate(painpoint_cols):
        prefs = customer_df[col].values
        scores = len(painpoint_cols) - i
        
        for pref in prefs:
            if pref in pref_scores:
                pref_scores[pref] += scores
                pref_counts[pref] += 1
            else:
                pref_scores[pref] = scores
                pref_counts[pref] = 1
    
    pref_data = pd.DataFrame(list(pref_scores.items()), columns=['painPointId', 'score'])
    pref_data['count'] = pref_data['painPointId'].map(pref_counts)
    pref_data['percentage'] = (pref_data['count'] / pref_data['count'].sum()) * 100

    # Get customerpainpoint and featurename from mf_content using painPointId
    pref_data = pref_data.merge(mf_content[['painPointId', 'customerPainPoint', 'featureName']], on='painPointId', how='left')
    
    sorted_pref_data = pref_data.sort_values('score', ascending=False)
    return sorted_pref_data

# Tab 1: Analytics - Visualize Customer Trends
def visualize_customer_trends(container = st):
            #opt1, opt2, opt3 = st.columns([1, 1, 1])
            #core_ref = opt1.selectbox("Display By", ["customerPainPoint", "featureName"], index=1)
            core_ref = "featureName"

            t1, t2, t3 = container.tabs(["Painpoint Metrics","Bar Chart", "Pie Chart"])
            
            # Tab 1: Painpoint Metrics
            t1.markdown("###### Painpoint Metrics")
            t1.data_editor(st.session_state.painpoint_metrics.head(7), hide_index=True, use_container_width=True, column_config=analytics_column_config)
            

            # Tab 2: Bar Chart
            painpoint_data = pd.DataFrame({
                'pain_point': st.session_state.painpoint_metrics[core_ref],
                'score': st.session_state.painpoint_metrics['score']
            })
            chart = alt.Chart(painpoint_data).mark_bar().encode(
                x=alt.X('score', title='Score'),
                y=alt.Y('pain_point', title='Customer Pain Point', sort='-x'),
                tooltip=['pain_point', 'score']
            ).properties(
                title='Top Features',
                width=600,
                height=500
            ).interactive()
            t2.altair_chart(chart, use_container_width=True)

            # Tab 3: Pie Chart
            painpoint_data = pd.DataFrame({
                'pain_point': st.session_state.painpoint_metrics[core_ref],
                'percentage': st.session_state.painpoint_metrics['percentage']
            })
            other_percentage = 100 - painpoint_data['percentage'].sum()
            painpoint_data = painpoint_data._append({'pain_point': 'Other', 'percentage': other_percentage}, ignore_index=True)
            chart = alt.Chart(painpoint_data).mark_arc().encode(
                theta=alt.Theta('percentage:Q', stack=True),
                color=alt.Color('pain_point:N', legend=alt.Legend(title='Pain Points')),
                tooltip=[alt.Tooltip('pain_point:N', title='Pain Point'), alt.Tooltip('percentage:Q', format='.1f', title='Percentage')]
            ).properties(
                title='Customer Pain Point Distribution',
                width=400,
                height=400
            )
            text = chart.mark_text(radius=140, size=12).encode(
                text=alt.Text('percentage:Q', format='.1f'),
                color=alt.value('white')
            )
            pie_chart = chart + text
            t3.altair_chart(pie_chart, use_container_width=True)

# Tab 2: View Logs - View Log Parquet
def view_log_parquet():
    # Try reading the Parquet file
    try:
        data = pd.read_parquet('assets/log.parquet')
        parquet_change = st.data_editor(data, num_rows='dynamic')
    except:
        # Error message if the file is not found
        st.error("Parquet file not found. Please generate dummy data first.")
    
    if st.button("Update Parquet"):
        # Convert parquet_change to an Arrow Table with the specified schema
        table = pa.Table.from_pandas(parquet_change, schema=parquet_schema_log)
        # Write the Arrow Table to the Parquet file
        pq.write_table(table, 'assets/log.parquet')

    if st.button("Generate Dummy Data"):
        generate_dummy_data('assets/log.parquet')
        st.success("Dummy data generated successfully!")

# Tab 2: View Logs - Generate Dummy Data
def generate_dummy_data(parquet_file):
    # Generate dummy data
    dummy_data = pd.DataFrame({
        "customer_name": np.random.choice(["John", "Jane", "Mike", "Emily"], size=100),
        "customer_title": np.random.choice(["Manager", "Director", "CXO"], size=100),
        "customer_company": np.random.choice(["Company A", "Company B", "Company C"], size=100),
        "persona_category1": np.random.choice(["Decision Maker", "Influencer", "Operational User"], size=100),
        "persona_category2": np.random.choice(["1-100 Employees", "101-500", "500+ Employees"], size=100),
        "persona_category3": np.random.choice(["Marketing Manager", "Software Engineer", "Product Manager"], size=100),
        "user_input": np.random.choice(["Input 1", "Input 2", "Input 3"], size=100),
        "paintpoints": [np.random.randint(1, 18, size=7) for _ in range(100)],
        "date": pd.to_datetime(np.random.choice(pd.date_range(start='2023-04-01', end='2024-04-01'), size=100)).strftime("%Y-%m-%d"),
        "time": pd.to_datetime(np.random.choice(pd.date_range(start='2022-01-01 10:00:00', end='2022-01-01 16:00:00', freq='s'), size=100)).strftime("%H:%M:%S")
    })

    # Convert the DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(dummy_data, preserve_index=False)

    if os.path.exists(parquet_file):
        # Read the existing data from the Parquet file
        existing_data = pq.read_table(parquet_file)

        # Concatenate the existing data with the new data
        combined_data = pa.concat_tables([existing_data, table])

        # Write the combined data back to the Parquet file
        pq.write_table(combined_data, parquet_file)
    else:
        # If the Parquet file is empty or doesn't exist, write the new data directly
        pq.write_table(table, parquet_file)

# Tab 3: Customer Personas
def update_customer_personas():
    st.subheader("Customer Personas")
    st.write("Upload a CSV file with columns category name, persona name, and persona description.")
    # Add a download button for template
    col1, col2 = st.columns([10, 1])
    template_csv = "assets/templates/cp_template.csv"
    with open(template_csv, "r") as file:
        bth = col2.download_button("Download Template", file, file_name="cp_template.csv", mime="text/csv")

    uploaded_file = col1.file_uploader("Upload CSV File:", label_visibility= 'collapsed', type=["csv"])

    if uploaded_file is not None:
        # Preview the uploaded file using st.write
        df = pd.read_csv(uploaded_file)
        edited_data = st.data_editor(df, hide_index=True, use_container_width=True)
        
        # Add a button to save the uploaded file as a json file
        if st.button("Upload Customer Personas"):
            # Save the edited data as a CSV file
            edited_data.to_csv("assets/customer_personas.csv", index=False)
            st.success("File saved successfully!")


## Session State Stuff
st.session_state.clicked = False
def click_button():
    st.session_state.display_metrics = True
if "display_metrics" not in st.session_state:
    st.session_state.display_metrics = False
if "painpoint_metrics" not in st.session_state:
    st.session_state.painpoint_metrics = None


## Streamlit code
# Setup
st.set_page_config(page_title="Analytics", page_icon=":bar_chart:", layout="wide")
st.header("Customer Analytics")
df, mf_content = get_mf_and_log(log_file = 'assets/log.parquet', mf_file = 'assets/mf_embeddings.parquet')

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Analytics", "View Logs", "Feature Leaderboard", "Customer Personas"])

with tab1:
    filter_col, content_col = st.columns([1, 3])
    filter = filter_col.container(border=True)
    content = content_col.container(border=True)
    title, date_range, persona_category1, persona_category2, persona_category3 = create_filter_components(df, filter)

    if st.button("Display Painpoint Metrics", on_click=click_button):
        if (title or persona_category1 or persona_category2 or persona_category3) is not None:
            # Get painpoint metrics for the selected filters
            st.session_state.painpoint_metrics = get_painpoint_metrics(df, mf_content, title, date_range, persona_category1, persona_category2, persona_category3)
        else:
            st.error("Please select at least one filter to display painpoint metrics.")
            
    # Display the DataFrame
    if st.session_state.display_metrics:
        content.markdown("#### Visualize Customer Trends")
        visualize_customer_trends(content)   

with tab2:
    # Code for the second tab
    st.title("View Logs")
    view_log_parquet()

with tab3:
    # Code for the third tab
    st.title("Feature Leaderboard")
    st.session_state.global_metrics = get_painpoint_metrics(df, mf_content, get_all=True)

with tab4:
    update_customer_personas()
