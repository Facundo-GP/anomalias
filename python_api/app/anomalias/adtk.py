from adtk.detector import SeasonalAD
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from anomalias import log

logger = log.logger('adtk')


class Adtk_AD:
    def __init__(self, model_type, **kargs):
        if model_type is 'SeasonalAD':
            logger.info('Creating SeasonalAD model.')
            self.__model = SeasonalAD()
        elif model_type is 'MinClusterAD':
            logger.info('Creating MinClusterAD model.')
            self.__model = MinClusterDetector(KMeans(**kargs))
        else:
            logger.error('Model type not found: %s', model_type)
            raise ValueError('Model type not found')

    def fit(self, train_data):
        logger.info('Fitting model.')
        self.__train_serie = train_data
        self.__model.fit(train_data)
        logger.info('Model fitted.')

    def detect(self, observations):
        idx_anom = self.__model.detect(observations)
        return idx_anom
