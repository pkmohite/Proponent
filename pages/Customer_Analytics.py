import pandas as pd
import streamlit as st
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from assets.code.element_configs import parquet_schema_log
import matplotlib.pyplot as plt
import textwrap
import altair as alt

def generate_dummy_data(parquet_file):
    # Generate dummy data
    dummy_data = pd.DataFrame({
        "customer_name": np.random.choice(["John", "Jane", "Mike", "Emily"], size=100),
        "customer_title": np.random.choice(["Manager", "Director", "Engineer"], size=100),
        "customer_company": np.random.choice(["Company A", "Company B", "Company C"], size=100),
        "persona_category1": np.random.choice(["Category 1", "Category 2", "Category 3"], size=100),
        "persona_category2": np.random.choice(["Category A", "Category B", "Category C"], size=100),
        "persona_category3": np.random.choice(["Category X", "Category Y", "Category Z"], size=100),
        "user_input": np.random.choice(["Input 1", "Input 2", "Input 3"], size=100),
        "paintpoints": [np.random.randint(1, 20, size=7) for _ in range(100)],
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "time": pd.Timestamp.now().strftime("%H:%M:%S")
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


def read_parquet(parquet_file):
    # Read the Parquet file into an Arrow Table
    table = pq.read_table(parquet_file)
    # Convert the Arrow Table to a Pandas DataFrame
    df = table.to_pandas()
    # Convert the 'paintpoints' array into separate columns
    df = pd.concat([df.drop('paintpoints', axis=1), df['paintpoints'].apply(pd.Series).add_prefix('pp')], axis=1)
    return df


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


def get_content():
    # Read the data from log.parquet
    df = read_parquet('assets/log.parquet')
    
    # Read the data from assets/mf_embeddings.parquet
    mf_content = pq.read_table('assets/mf_embeddings.parquet').to_pandas()

    return df, mf_content


def create_filter_components(df):
    a1, a2, a3 = st.columns([2, 1, 1])
    b1, b2, b3 = st.columns([1, 1, 1])

    title = a1.selectbox("Customer Title", df['customer_title'].unique(), index=None)
    from_date = a2.date_input("From Date", None)
    to_date = a3.date_input("To Date", None)
    date_range = [from_date, to_date]
       

    persona_category1 = b1.selectbox("Persona Category 1", df['persona_category1'].unique(), index=None)
    persona_category2 = b2.selectbox("Persona Category 2", df['persona_category2'].unique(), index=None)
    persona_category3 = b3.selectbox("Persona Category 3", df['persona_category3'].unique(), index=None)

    return title, date_range, persona_category1, persona_category2, persona_category3


def click_button():
    st.session_state.display_metrics = True

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



def visualize_customer_trends():
            opt1, opt2, opt3 = st.columns([1, 1, 1])
            core_ref = opt1.selectbox("Display By", ["customerPainPoint", "featureName"], index=1)


            t1, t2, t3 = st.tabs(["Bar Chart", "Pie Chart", "Altair Chart"])
            with t1.empty():
                
                painpoint_data = pd.DataFrame({
                    'pain_point': st.session_state.painpoint_metrics[core_ref],
                    'score': st.session_state.painpoint_metrics['score']
                })

                # Create the Altair chart
                chart = alt.Chart(painpoint_data).mark_bar().encode(
                    x=alt.X('score', title='Score'),
                    y=alt.Y('pain_point', title='Customer Pain Point', sort='-x'),
                    tooltip=['pain_point', 'score']
                ).properties(
                    title='Top 7 Customer Pain Points',
                    width=600,
                    height=500
                ).interactive()

                # Display the chart using Streamlit
                st.altair_chart(chart, use_container_width=True)

            with t2.empty():
                painpoint_data = pd.DataFrame({
                    'pain_point': st.session_state.painpoint_metrics[core_ref],
                    'percentage': st.session_state.painpoint_metrics['percentage']
                })

                # Calculate the "other" percentage
                other_percentage = 100 - painpoint_data['percentage'].sum()

                # Add the "other" category to the DataFrame
                painpoint_data = painpoint_data._append({'pain_point': 'Other', 'percentage': other_percentage}, ignore_index=True)

                # Create the Altair chart
                chart = alt.Chart(painpoint_data).mark_arc().encode(
                    theta=alt.Theta('percentage:Q', stack=True),
                    color=alt.Color('pain_point:N', legend=alt.Legend(title='Pain Points')),
                    tooltip=[alt.Tooltip('pain_point:N', title='Pain Point'), alt.Tooltip('percentage:Q', format='.1f', title='Percentage')]
                ).properties(
                    title='Customer Pain Point Distribution',
                    width=400,
                    height=400
                )

                # Add percentage labels to the chart
                text = chart.mark_text(radius=140, size=12).encode(
                    text=alt.Text('percentage:Q', format='.1f'),
                    color=alt.value('white')
                )

                # Combine the chart and text labels
                pie_chart = chart + text

                # Display the chart using Streamlit
                st.altair_chart(pie_chart, use_container_width=True)



## Session State Stuff
if "display_metrics" not in st.session_state:
    st.session_state.display_metrics = False
if "painpoint_metrics" not in st.session_state:
    st.session_state.painpoint_metrics = None

## Streamlit code
# Setup
st.set_page_config(page_title="Analytics", page_icon=":bar_chart:", layout="wide")
st.title("Customer Analytics")
df, mf_content = get_content()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Analytics", "View Logs", "Feature Leaderboard", "Customer Personas"])

with tab1:
    title, date_range, persona_category1, persona_category2, persona_category3 = create_filter_components(df)

    if st.button("Display Painpoint Metrics", on_click=click_button):
        if (title or date_range or persona_category1 or persona_category2 or persona_category3) is not None:
            # Get painpoint metrics for the selected filters
            st.session_state.painpoint_metrics = get_painpoint_metrics(df, mf_content, title, date_range, persona_category1, persona_category2, persona_category3)
        else:
            st.error("Please select at least one filter to display painpoint metrics.")
            
    # Display the DataFrame
    if st.session_state.display_metrics:
        st.markdown("#### Painpoint Metrics")
        st.data_editor(st.session_state.painpoint_metrics.head(7), hide_index=True, use_container_width=True)

        st.markdown("#### Visualize Customer Trends")
        visualize_customer_trends()   


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
