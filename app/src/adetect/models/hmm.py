import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
import scipy.stats

from anomalias import log

logger = log.logger('adtk')


class HiddenMarkovModel_AD:

    def __init__(self, **kwargs):
        
        logger.info('Creating Hidden Markov Model.')
        self.__n_components = 3
        self.__model = hmm.GaussianHMM(n_components=3, n_iter=100)
        self.__distributions = {}
        self.__anomalies_states = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        logger.info('Fitting model.')
        
        #Entrenamiento
        self.__model.fit(X.values)

        #Obtencion de distribuciones
        for index,stats in enumerate(zip(self.__model.means_, self.__model.covars_)):
            self.__distributions[index] = { "pdf" : scipy.stats.norm(stats[0][0],np.sqrt(stats[1][0][0])), "samples": 0}

        #Clasificacion de estados anomalos
        for pred in self.__model.predict(X):
            self.__distributions[pred]["samples"]+=1

        total = len(X)
        for key in list(self.__distributions.keys()):
            prop = ( (self.__distributions[key]["samples"] / total ) * 100 )
            if prop < 30 / self.__n_components:
                self.__anomalies_states.append(key)

        logger.info('Model fitted.')

    def detect(self, observations: pd.DataFrame):
        
        self.__detected_anomalies = []
        for obs,pred in zip(observations.values,self.__model.predict(observations.values)):
            
            expected_value = self.__distributions[pred]["pdf"].mean()
            std = self.__distributions[pred]["pdf"].std()
            
            if obs >= 3*std + expected_value or obs <= expected_value-3*std:
                self.__detected_anomalies.append(True)
            elif pred in self.__anomalies_states:
                self.__detected_anomalies.append(True)
            else:
                self.__detected_anomalies.append(False)

        return pd.Series(data=self.__detected_anomalies,index=observations.index)