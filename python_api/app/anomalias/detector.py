import pandas as pd
import anomalias.log as log
from threading import Condition
from anomalias.tsmodels import SSM_AD
from anomalias.adtk import Adtk_AD

logger = log.logger('Detector')


class Detector:
    def __init__(self, len):
        # Series
        self.__len = len
        self.__available = Condition()
        self.__dataFrame = pd.DataFrame([])
        self.__anomalies = pd.DataFrame([], dtype='boolean')

        self.__training = False
        self.__paused = False

    def ssm_ad(self, th, endog, model_type, **kwargs):
        with self.__available:
            # Model
            logger.info('Creating Anomaly Detector.')
            self.__model = SSM_AD(th, endog, model_type, **kwargs)
            self.__available.notify()

    def adtk_ad(self, model_type, **kargs):
        with self.__available:
            self.__model = Adtk_AD(model_type, **kargs)
            self.__available.notify()

    def fit(self, serie):
        with self.__available:
            self.__model.fit(serie)
            self.__available.notify()

    def detect(self, observations):
        with self.__available:
            # Series Update
            dataFrame = observations[~observations.index.isin(self.__dataFrame.index)]
            self.__dataFrame = pd.concat([self.__dataFrame, dataFrame]).iloc[-self.__len:]
            # Detection
            anomalies = self.__model.detect(dataFrame).astype('boolean')
            self.__anomalies = pd.concat([self.__anomalies, anomalies]).iloc[-self.__len:]
            self.__available.notify()

            return dataFrame, anomalies

    def get_detection(self):
        return self.__dataFrame.copy(), self.__anomalies.copy()