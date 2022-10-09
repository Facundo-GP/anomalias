import numpy as np
import pandas as pd


def point_anomaly(df: pd.DataFrame, anomaly_rate: float = 0.001, unilateral: bool = True)\
        -> pd.DataFrame:
    """ Generate point anomalies """
    out_df = df.copy()

    # Select column at random
    col = np.random.choice(out_df.columns)

    # Select indexes at random
    indexes = np.random.choice(out_df.index, size=int(len(out_df) * anomaly_rate), replace=False)

    # Generate anomalies
    if unilateral:
        # Unilateral anomaly
        out_df.loc[indexes, col] = out_df[col].max() * (1 + np.random.rand(len(indexes))*2)
    else:
        raise NotImplementedError

    # Add label column
    out_df['label'] = False
    out_df.loc[indexes, 'label'] = True

    return out_df
