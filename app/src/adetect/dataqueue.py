import queue
from threading import Thread
import pandas as pd

from . import detector
from . import log
from .api import InfluxApi

logger = log.logger('Series')


class DataQueue(Thread):
    def __init__(self, id: str, len: int, api: InfluxApi):
        super(Thread, self).__init__(self, name=id)
        self.id = id
        self.ad = detector.Detector(len=len)

        self.__exit = False
        self.__paused = False
        self.__observations = queue.Queue()
        self.__api = api

        logger.info('New series created, id %s', self.id)

    def run(self):
        while not self.__exit:
            obs = self.__observations.get()
            if not obs.empty:
                data, anomalies = self.ad.detect(obs)

                if isinstance(anomalies, pd.Series):
                    anomalies = anomalies.to_frame()[[0] * data.shape[-1]]
                    anomalies.columns = data.columns

                logger.debug('Data for detection:')
                logger.debug('\n %s', data)
                logger.debug('Anomalies:')
                logger.debug('\n %s', anomalies)

                self.__api.write(data, anomalies, self.__obs_name)

    def append(self, obs, obs_name):
        self.__observations.put(obs)
        self.__obs_name = obs_name
        self.run()

    def exit(self, bol=False):
        self.__exit = bol
        self.__api.close()

    def pause(self):
        self.__paused = True

    def resume(self):
        self.__paused = False


