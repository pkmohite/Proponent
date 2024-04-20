import streamlit as st
from assets.code.utils import check_password, set_page_config, get_themed_logo
import os

# Setup
set_page_config(page_title="Proponent", layout="centered",initial_sidebar_state="collapsed")

if not check_password():
    st.stop()
else:
    st.switch_page("pages/1_Home.py")