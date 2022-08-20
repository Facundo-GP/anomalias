import anomalias.log as log
import anomalias.dataframe as dataframe

logger = log.logger('anomd')


class Detectors():
    def __init__(self):
        self.__dataframes = []
        self.df_id = []

    def add(self, id, len, api):
        try:
            self.__dataframes.append(dataframe.DataFrame(id=id, len=len, api=api))
            self.df_id.append(id)
        except Exception as e:
            logger.error('%s', e)
            raise

    def ssm_ad(self, id, th, endog, model_type, **kwargs):
        try:
            if self.__exist_id(id):
                (self.__dataframes[self.df_id.index(id)]).ad.ssm_ad(th, endog, model_type, **kwargs)
        except Exception as e:
            logger.error('%s', e)
            return None

    def adtk_ad(self, id, model_type, **kwargs):
        try:
            if self.__exist_id(id):
                (self.__dataframes[self.df_id.index(id)]).ad.adtk_ad(model_type, **kwargs)
        except Exception as e:
            logger.error('%s', e)
            return None

    def remove(self, id):
        try:
            if self.__exist_id(id):
                index = self.df_id.index(id)
                self.__dataframes[index].exit()
                del self.__dataframes[index]
                self.df_id.remove(id)
        except Exception as e:
            logger.error('%s', e)
            return None

    def list_id(self):
        return self.df_id

    def start(self, id):
        try:
            if self.__exist_id(id):
                if not self.__dataframes[self.df_id.index(id)].is_alive():
                    self.__dataframes[self.df_id.index(id)].start()
                else:
                    logger.warning('Series is running, id: %s', id)
        except Exception as e:
            logger.error('%s', e)

    def append(self, id, obs, obs_name):
        if self.__exist_id(id):
            self.__dataframes[self.df_id.index(id)].append(obs, obs_name)

    def fit(self, id, dataFrame):
        if self.__exist_id(id):
            self.__dataframes[self.df_id.index(id)].ad.fit(dataFrame)

    def get_detection(self, id):
        try:
            if self.__exist_id(id):
                if self.__dataframes[self.df_id.index(id)].isAlive():
                    df, anom = (self.__dataframes[self.df_id.index(id)]).ad.get_detection()
                    return df, anom
                else:
                    logger.info('Time series is not running, id: %s', id)
                    return None
        except Exception as e:
            logger.error('%s', e)
            return None

    def __exist_id(self, id):
        try:
            if self.df_id.__contains__(id):
                return True
            else:
                logger.warning('Time series not found, id: %s', id)
                return False
        except Exception as e:
            logger.error('%s', e)
            return False
