Machine Learning ETL Pipeline with Cloud Deployment

An end-to-end data pipeline that integrates ETL, feature engineering, model training, and cloud deployment for credit risk prediction.  
The project demonstrates how data engineering and machine learning techniques work together to build scalable, governed, and production-style analytics systems.

Overview
The dataset includes both categorical and numeric variables representing customer financial profiles.  
The pipeline performs data extraction, cleaning, and transformation, followed by model training to predict credit default risk.  
The trained model is deployed using Streamlit Cloud for interactive inference and visualization.

Process
- Extracts raw CSV data and validates missing or inconsistent records.
- Performs feature engineering and preprocessing for numeric and categorical fields.
- Trains multiple models (Logistic Regression, Random Forest, XGBoost) and evaluates with ROC-AUC.
- Saves the best-performing model for deployment and version tracking.
- Deploys the model to Streamlit Cloud for public interaction.
- Documents the ETL, training, and deployment steps for full transparency.

Tools and Technologies
Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib, GitHub

Key Highlights
- Combines data engineering, model development, and deployment in a single pipeline.
- Implements lightweight governance through logging and versioned model storage.
- Demonstrates applied machine learning for financial risk management.
- Built to align with enterprise-level data quality and reproducibility standards.

Future Improvements
- Integrate MLflow or DVC for model tracking.
- Add Power BI dashboard for live performance monitoring.
- Expand to real-world credit risk datasets with additional features.


