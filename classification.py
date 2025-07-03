import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def classification_module(df):
    st.header("Classification")
    # Prepare data
    X = df.select_dtypes(include=['int64','float64']).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df['TryNewBrand'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }
    results = []
    cm_results = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
        cm_results[name] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

    results_df = pd.DataFrame(results)
    st.subheader("Metrics Comparison")
    st.dataframe(results_df)

    # Confusion matrix selector
    sel = st.selectbox("Select algorithm for confusion matrix", list(models.keys()))
    cm = cm_results[sel]
    st.subheader(f"Confusion Matrix: {sel}")
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=name)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Upload new data and predict
    st.subheader("Predict on New Data")
    uploaded = st.file_uploader("Upload CSV without target", type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        new_X = new_df[X.columns]
        preds = models[sel].predict(new_X)
        new_df['Prediction'] = LabelEncoder().inverse_transform(preds)
        st.write(new_df)
        csv = new_df.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv, file_name="predictions.csv")
