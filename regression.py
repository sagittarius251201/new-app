import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def regression_module(df):
    st.header("Regression Insights")
    X = df[['MonthlyDisposableIncome', 'HealthConsciousness', 'Age']]
    y = df['SpendPerServing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        results.append({
            'Model': name,
            'Train R2': r2_score(y_train, pred_train),
            'Test R2': r2_score(y_test, pred_test),
            'Train RMSE': mean_squared_error(y_train, pred_train, squared=False),
            'Test RMSE': mean_squared_error(y_test, pred_test, squared=False)
        })
    st.subheader("Regression Results")
    st.dataframe(pd.DataFrame(results))
