# Health Drink Survey Dashboard

This Streamlit app provides comprehensive analysis for a synthetic consumer survey dataset for a health drink launched in the UAE market, including:

- Data Visualization
- Classification (KNN, Decision Tree, Random Forest, GBRT)
- Clustering (KMeans)
- Association Rule Mining (Apriori)
- Regression (Linear, Ridge, Lasso, Decision Tree Regressor)

## Getting Started

1. Clone the repo to your local machine.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your GitHub repository and push the code.
4. Connect to Streamlit Cloud and deploy the app:
   - In Streamlit Cloud, select "Deploy from GitHub repo".
   - Provide the repository URL.
   - Ensure Python version is 3.8+.
5. Ensure your dataset CSV (`health_drink_survey_1000_responses.csv`) is placed in the root or `data/` folder of the repo.
   - Alternatively, use the GitHub raw file URL in the app.

## Usage

Run locally:
```
streamlit run app.py
```

## File Structure

```
health_drink_dashboard/
├── app.py
├── src/
│   ├── data_loader.py
│   ├── visualizations.py
│   ├── classification.py
│   ├── clustering.py
│   ├── association.py
│   ├── regression.py
├── requirements.txt
└── README.md
```
