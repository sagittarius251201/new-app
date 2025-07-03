import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.visualizations import show_visualizations
from src.classification import classification_module
from src.clustering import clustering_module
from src.association import association_module
from src.regression import regression_module

st.set_page_config(layout="wide", page_title="Health Drink Dashboard")
st.title("Health Drink Survey Dashboard")

# Data source selection
st.sidebar.header("Data Source")
data_option = st.sidebar.radio("Choose source:", ["GitHub URL", "Upload CSV"])
if data_option == "GitHub URL":
    url = st.sidebar.text_input("Enter raw GitHub CSV URL")
    if url:
        df = load_data(url)
    else:
        st.warning("Please enter a valid URL.")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Main tabs
tabs = st.tabs(["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"])

with tabs[0]:
    show_visualizations(df)
with tabs[1]:
    classification_module(df)
with tabs[2]:
    clustering_module(df)
with tabs[3]:
    association_module(df)
with tabs[4]:
    regression_module(df)
