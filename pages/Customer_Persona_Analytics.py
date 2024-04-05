import pandas as pd
import streamlit as st
import os

# Check if the file exists
if os.path.exists('assets/log.parquet'):
    # Read the Parquet file
    data = pd.read_parquet('assets/log.parquet')

    # Display the data using st.write
    st.data_editor(data)
else:
    st.write("File 'assets/log.parquet' does not exist.")

