"""
Time Series Models
"""
import statsmodels.tsa.statespace.api as ss
from anomalias import log
import numpy as np

logger = log.logger( 'tsmodels' )

class SSM_AD:
    def __init__(self, th, endog, model_type, params=None, **kwargs):
        if model_type is 'SARIMAX':
            logger.info('Creating SARIMAX model.')
            self.__train_serie = endog
            self.__th = th
            self.__model = ss.SARIMAX(self.__train_serie, **kwargs)
            if params is None:
                self.__model_fit = self.__model.fit()
                logger.info('Model fitted. \n %s', self.__model_fit.summary())
            else:
                self.__model.update(params=params)
                self.__model_fit = self.__model.filter()
        else:
            logger.error('Model type not found: %s', model_type)
            raise ValueError('Model type not found')

    def train(self, train_data, params=None):
        logger.info('Fitting model...')
        self.__train_serie = train_data
        if params is None:
            self.__model_fit.apply(endog=train_data, refit=True)
            logger.info('Model fitted. \n %s', self.__model_fit.summary())
        else:
            self.__model.update(params=params)
            self.__model_fit.apply(endog=train_data)
        logger.info('Model fitted. \n %s', self.__model_fit.summary())

    def detect(self, observations):
        self.__model_fit = self.__model_fit.extend(observations)
        pred = self.__model_fit.get_prediction()

        prediction_error = observations - pred.predicted_mean
        sigma = np.sqrt(pred.var_pred_mean)
        idx_anom = np.abs(prediction_error) > self.__th * sigma

        return idx_anom



