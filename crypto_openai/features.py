import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute return, MA7, MA30, volatility (écart-type des returns sur 30 périodes)."""
    df = df.copy()
    df["return"] = df["price"].pct_change().fillna(0.0)
    df["MA7"] = df["price"].rolling(7, min_periods=1).mean()
    df["MA30"] = df["price"].rolling(30, min_periods=1).mean()
    # volatilité = std des returns sur 30 périodes (relative)
    df["volatility"] = df["return"].rolling(30, min_periods=1).std().fillna(0.0)
    return df
