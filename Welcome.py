import streamlit as st
from assets.code.utils import check_password

if not check_password():
    st.stop()