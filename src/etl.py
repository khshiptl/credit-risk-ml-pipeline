from pathlib import Path
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p)

def save_parquet(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

