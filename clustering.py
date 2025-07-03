import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def clustering_module(df):
    st.header("Clustering")
    # Select features
    features = ['Age', 'MonthlyDisposableIncome', 'SpendPerServing', 'HealthConsciousness']
    data = df[features]

    # Elbow chart
    st.subheader("Elbow Chart")
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        sse.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(k_range, sse, marker='o')
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("SSE")
    st.pyplot(fig)

    # Cluster slider
    k = st.slider("Select number of clusters", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = km.fit_predict(data)

    # Persona table
    st.subheader("Cluster Personas")
    persona = df.groupby('Cluster')[features].mean().round(2)
    st.dataframe(persona)

    # Download full data
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Clustered Data", csv, file_name="clustered_data.csv")
