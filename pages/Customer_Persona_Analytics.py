import pandas as pd
import streamlit as st
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from element_configs import parquet_schema

# Try reading the Parquet file
try:
    data = pd.read_parquet('assets/log.parquet')
    parquet_change = st.data_editor(data, num_rows='dynamic')
except:
    # Error message if the file is not found
    st.error("Parquet file not found. Please generate dummy data first.")

if st.button("Update Parquet"):
    # Convert parquet_change to an Arrow Table with the specified schema
    table = pa.Table.from_pandas(parquet_change, schema=parquet_schema)
    # Write the Arrow Table to the Parquet file
    pq.write_table(table, 'assets/log.parquet')

if st.button("Generate Dummy Data"):
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

    parquet_file = "assets/log.parquet"
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
