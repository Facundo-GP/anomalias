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


def resample_data(df: pd.DataFrame | pd.Series, method: str = 'ffill'):
    if method == 'ffill':
        return df.resample('5T').ffill().fillna(0).astype(int)
    else:
        raise NotImplementedError


def gen_drop_anomaly(data: pd.Series, split_factor: float = 0.6, window_size: int = 1001, plot: bool = false):
    """
    Generate drop off anomaly on the test portion of the time series
    :param data: Data, must be indexed by time-date
    :param split_factor: Ratio of samples assigned to training, starting at the beginning of the series
    :param window_size: Size of the window used to generate the anomaly, in time steps
    :param plot: If True, plot figure with result
    :return: Dataframe with the anomalous series and anomaly labels
    """
    assert isinstance(data.index, pd.DatetimeIndex), 'Index must be of type `datetime`'

    test_start_idx = int(len(data)*split_factor)
    test_start_dt = data.index[test_start_idx]

    window = 1 - np.hamming(window_size)
    start = np.random.randint(low=test_start_idx, high=len(data)-window_size)
    indexes = data.index[start:start+window_size]

    anomaly_mask = pd.Series(index=data.index, data=1, name='anomaly', dtype=float)
    anomaly_mask[indexes] = window

    df = pd.DataFrame(index=data.index, data={'data': data * anomaly_mask, 'anomaly': -(anomaly_mask-1) > 0.1})

    if plot:
        pass
        # TODO

    return df
