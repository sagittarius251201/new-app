import pandas as pd
import streamlit as st

@st.cache_data
def load_data(csv_url):
    return pd.read_csv(csv_url)
