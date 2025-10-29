from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.features import split_features_target, build_preprocessor, train_val_split

def train_model(df: pd.DataFrame, target: str, model_dir: str):
    X, y = split_features_target(df, target)
    pre = build_preprocessor(X)
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p)
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump(pipe, model_path)
    return {"auc": auc, "model_path": str(model_path)}

if __name__ == "__main__":
    import pandas as pd
    from src.etl import load_csv

    df = load_csv("data/raw/credit_data.csv")
    result = train_model(df, target="default", model_dir="models")
    print(f"Model trained. AUC: {result['auc']:.3f}")
    print(f"Saved model to: {result['model_path']}")

