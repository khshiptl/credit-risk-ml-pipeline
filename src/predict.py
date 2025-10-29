from pathlib import Path
import joblib
import pandas as pd

def load_model(path: str):
    return joblib.load(Path(path))

def predict_proba(model, df: pd.DataFrame):
    return model.predict_proba(df)[:, 1]

