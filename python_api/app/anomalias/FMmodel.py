from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

import log
# TODO install package
from polylearn import FactorizationMachineRegressor

logger = log.logger('FMmodel')


class FactorizationMachineAnomalyDetector:
    """
    Factorization Machine Anomaly Detector
    """
    def __init__(self, window_size: int, fm_params: dict):
        logger.info('Creating FM model.')
        self.__pipe = make_pipeline([
            PreprocessingFMTransformer(window_size),
            FactorizationMachineRegressor(**fm_params)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info('Fitting model...')
        self.__pipe.fit(X, y)
        logger.info(f'Model fitted.\n')

    def detect(self, observations: pd.DataFrame) -> pd.Series:
        logger.info('Detecting anomalies...')
        # TODO decide logic for classification of anomalies
        return self.__pipe.predict(observations)


class PreprocessingFMTransformer(BaseEstimator, TransformerMixin):
    """
    Preprocess time series data to use on FM model
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        logger.debug('Creating preprocessing FM transformer.')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        assert X.index.dtype == datetime, 'Data must be a pandas DataFrame with datetime index.'
        if y is not None:
            assert y.index.dtype == datetime, 'Data must be a pandas DataFrame with datetime index.'
            assert X.index.equals(y.index), 'Data must have the same index.'
            assert y.name in X.columns, 'Target must be a column of data.'
        return self

    def transform(self, X: pd.Dataframe, y: pd.Series = None):
        out_values = np.empty(((len(X) - self.window_size),
                               X.shape[1]*self.window_size + 4))  # +4 is for date-time data

        # Process values on each time series
        for col in X:
            out_col_values = np.hstack(col.values[i:-self.window_size+i] for i in range(self.window_size))
            out_values = np.hstack((out_values, out_col_values))

        # Process date-time data
        dt_values = split_datetime(X.index).values
        out_values = np.hstack((out_values, dt_values[self.window_size:, :]))

        return out_values


def split_datetime(dt_series: pd.Series) -> pd.DataFrame:
    """ Split datetime data into separate columns """
    out_df = pd.DataFrame()
    out_df['day'] = dt_series.dt.day
    out_df['weekday'] = dt_series.dt.weekday
    out_df['month'] = dt_series.dt.month
    out_df['hour'] = dt_series.dt.hour

    return out_df


if __name__ == '__main__':
    test_data = pd.DataFrame({'serie': np.random.randn(100)},
                             index=pd.date_range('1/1/2000', periods=100, freq='5min'))

    fm_model = FactorizationMachineAnomalyDetector(window_size=5, fm_params={'n_iter': 10})

    fm_model.train(test_data, test_data['serie'])