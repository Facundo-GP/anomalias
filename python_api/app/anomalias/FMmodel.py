import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

import log
from fastFM.als import FMRegression

logger = log.logger('FMmodel')

class FactorizationMachineAnomalyDetector:
    """
    Factorization Machine Anomaly Detector
    """
    def __init__(self, target: str, anomaly_type: str, window_size: int, fm_params: dict, **kwargs):
        logger.info('Creating FM model.')
        if anomaly_type == 'point':
            self.anomaly_type = anomaly_type
            try:
                self.threshold = kwargs['threshold']
            except KeyError:
                logger.error('Threshold must be provided for point anomaly detection.')
        self.target = target
        self.__pipe = make_pipeline(
            PreprocessingFMTransformer(window_size),
            FMRegression(**fm_params)
        )

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info('Fitting model...')
        # First {window_size} targets are not used for training (sklearn pipelines won't change target data)
        self.__pipe.fit(X, y.iloc[self.__pipe['preprocessingfmtransformer'].window_size:])
        logger.info(f'Model fitted.\n')

    def detect(self, observations: pd.DataFrame) -> pd.Series:
        assert self.target in observations.columns, f'{self.target} not in observations columns.'
        logger.info('Detecting anomalies...')
        out_vals = np.zeros(len(observations))
        y_true = observations[self.target].values[self.__pipe['preprocessingfmtransformer'].window_size:]
        y_pred = self.__pipe.predict(observations).values
        out_vals[self.__pipe['preprocessingfmtransformer'].window_size:] = np.abs(y_true-y_pred) < self.threshold
        return pd.Series(out_vals, index=observations.index, name='anomalias', dtype=bool)


class PreprocessingFMTransformer(BaseEstimator, TransformerMixin):
    """
    Preprocess time series data to use on FM model
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        logger.debug('Creating preprocessing FM transformer.')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        assert isinstance(X.index, pd.DatetimeIndex), 'Data must be a pandas DataFrame with datetime index.'
        assert len(X) >= self.window_size, 'Data must have at least {self.window_size} rows.'
        if y is not None:
            assert isinstance(y.index, pd.DatetimeIndex), 'Data must be a pandas DataFrame with datetime index.'
            assert X.index[self.window_size:].equals(y.index), 'Data must have the same index.'
            assert y.name in X.columns, 'Target must be a column of data.'
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        # out_values = np.empty(((len(X) - self.window_size),
        #                        X.shape[1]*self.window_size + 4))  # +4 is for date-time data

        # Process date-time data
        out_values = split_datetime(X.index).values[self.window_size:, :]
        # out_values = np.hstack((out_values, dt_values[self.window_size:, :]))

        logger.debug(f'Transformed date values:\n'
                     f'{out_values}\n'
                     f'Shape: {out_values.shape}')

        # Process values on each time series
        for col in X:
            out_col_values = np.hstack((X[col].values[i:-self.window_size+i].reshape(-1, 1)
                                        for i in range(self.window_size)))

            out_values = np.hstack((out_values, out_col_values))

        logger.debug(f'Transformed column values:\n'
                     f'{out_values}\n'
                     f'Shape: {out_col_values.shape}')

        # fastFM requires sparse matrix
        return csc_matrix(out_values)


def split_datetime(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """ Split datetime data into separate columns """
    out_df = pd.DataFrame()
    out_df['day'] = dt_index.day
    out_df['weekday'] = dt_index.weekday
    out_df['month'] = dt_index.month
    out_df['hour'] = dt_index.hour

    return out_df


if __name__ == '__main__':
    test_data = pd.DataFrame({'serie': np.random.randn(100)},
                             index=pd.date_range('1/1/2000', periods=100, freq='5min'))

    logger.debug(f'{test_data}')

    fm_model = FactorizationMachineAnomalyDetector(anomaly_type='point',
                                                   threshold=0.01,
                                                   window_size=5,
                                                   fm_params={'n_iter': 10, 'rank': 2},
                                                   target='serie')

    fm_model.train(test_data, test_data['serie'])

    print(fm_model.detect(test_data))