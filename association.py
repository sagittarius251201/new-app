import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def association_module(df):
    st.header("Association Rule Mining")
    # Prepare boolean dataframe
    cols = [c for c in df.columns if c.startswith('Flavour_') or c.startswith('Context_')]
    subset = df[cols]
    min_support = st.slider("Min Support", 0.01, 0.1, 0.02)
    min_confidence = st.slider("Min Confidence", 0.1, 0.5, 0.3)
    itemsets = apriori(subset, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    st.subheader("Top 10 Rules by Lift")
    st.write(rules.sort_values('lift', ascending=False).head(10))
