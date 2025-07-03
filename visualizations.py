import streamlit as st
import matplotlib.pyplot as plt

def show_visualizations(df):
    # 1. Age distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Age'], bins=20)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 2. Income distribution
    st.subheader("Monthly Disposable Income Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['MonthlyDisposableIncome'], bins=20)
    ax.set_xlabel("Income (AED)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 3. Spend per serving distribution
    st.subheader("Spend per Serving Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['SpendPerServing'], bins=20)
    ax.set_xlabel("Spend (AED)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 4. Gender vs Spend boxplot
    st.subheader("Gender vs Spend per Serving")
    fig, ax = plt.subplots()
    df.boxplot(column='SpendPerServing', by='Gender', ax=ax)
    ax.set_ylabel("Spend (AED)")
    st.pyplot(fig)

    # 5. Occupation counts
    st.subheader("Occupation Counts")
    fig, ax = plt.subplots()
    df['Occupation'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Occupation")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 6. Health consciousness distribution
    st.subheader("Health Consciousness Levels")
    fig, ax = plt.subplots()
    df['HealthConsciousness'].value_counts(sort=False).plot(kind='bar', ax=ax)
    ax.set_xlabel("Level")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 7. Consumption frequency
    st.subheader("Health Drink Consumption Frequency")
    fig, ax = plt.subplots()
    df['ConsumptionFrequency'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 8. Top health benefits
    st.subheader("Top Health Benefits Sought")
    fig, ax = plt.subplots()
    df['TopHealthBenefit'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Benefit")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 9. Packaging format counts
    st.subheader("Preferred Packaging Formats")
    fig, ax = plt.subplots()
    df['PackagingFormat'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Format")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 10. Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=['int64','float64']).corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)
