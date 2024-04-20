import streamlit as st
from assets.code.utils import check_password, set_page_config, get_themed_logo
import os

os.environ['LOGIN'] = '{"test-username": "test-password", "user1": "password1", "user2": "password2", "user3": "password3"}'

# Setup
set_page_config(page_title="Proponent", page_icon=":wave:", layout="centered",initial_sidebar_state="collapsed")

if not check_password():
    st.stop()
else:
    st.switch_page("pages/1_Home.py")