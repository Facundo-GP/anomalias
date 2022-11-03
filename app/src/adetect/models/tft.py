import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss, NormalDistributionLoss, QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, DeepAR
import torch

import pandas as pd
import numpy as np


class TemporalFusionTransformer_AN():
    
    def __init__(self,values,factors):
        self.values = values
        self.factors = factors
        return

    def __create_time_idx(self,df: pd.DataFrame, values : list, series_factors: str, max_encoder_length: int,
                    max_prediction_length: int) -> pd.DataFrame:

        series = df[series_factors].unique()
        series_values = df[series_factors].values
        
        timeidx_dict = dict((serie, 0) for serie in series)
        series_timeidx = np.zeros(len(series_values), dtype=int)
        
        for i in range(len(series_timeidx)):
            series_timeidx[i] = timeidx_dict[series_values[i]]
            timeidx_dict[series_values[i]]+=1
        
        data = df.copy()
        data["time_idx"] = series_timeidx
        
        training = TimeSeriesDataSet(
        data,
        time_idx= "time_idx",
        target= values,
        group_ids= [series_factors],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals= [series_factors],
        time_varying_unknown_reals= [values],
        )


        return training, data

    def __create_trainer(self,epochs : int) -> pl.Trainer:
    
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=0,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches=50,
            enable_checkpointing=True,
        )

        return trainer


    def __make_predictions(self,model, data, prediction_lenght):
        predictions = model.predict(data, mode="quantiles")

        min_quantile_matrix = np.zeros((predictions.shape[0],prediction_lenght)) + 1e10
        max_quantile_matrix = np.zeros((predictions.shape[0],prediction_lenght)) - 1e10

        for bucket_index in range(len(predictions)):
            for time_index in range(len(predictions[bucket_index])):
                for data_column in range(min_quantile_matrix.shape[1]):
                    try:
                        if min_quantile_matrix[bucket_index + time_index][data_column] == 1e10:   
                                min_quantile_matrix[bucket_index + time_index][data_column] = predictions[bucket_index][time_index][0]
                                break
                    except:
                        break


        for bucket_index in range(len(predictions)):
            for time_index in range(len(predictions[bucket_index])):
                for data_column in range(max_quantile_matrix.shape[1]):
                    try:
                        if max_quantile_matrix[bucket_index + time_index][data_column] == -1e10:
                            max_quantile_matrix[bucket_index + time_index][data_column] = predictions[bucket_index][time_index][-1]
                            break
                    except:
                        break


        return np.column_stack((min_quantile_matrix.min(axis=1), max_quantile_matrix.max(axis=1)))


    def __detect_anomalies(self,data, quantiles):
        
        anomalies = np.zeros(len(data),dtype=bool)
        offset = len(data) - len(quantiles)

        for index,quantil in enumerate(quantiles):
            if data[index+offset] < quantil[0] or data[index+offset] > quantil[1]:
                anomalies[index+offset] = True

        return pd.Series(data=anomalies,index=data.index)


    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        
        self.max_encoder_length = 10
        self.max_prediction_length = 3
    
        training, data = self.__create_time_idx(X, self.values, self.factors,self.max_encoder_length,self.max_prediction_length)

        # create validation and training dataset
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
        batch_size = 32
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)
        

        # create the model
        net = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,
            loss=QuantileLoss(),

        )

        epochs = 1
        trainer = self.__create_trainer(epochs)
        



        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        best_model_path = trainer.checkpoint_callback.best_model_path
        self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


    def detect(self, observations: pd.DataFrame):

        _, data = self.__create_time_idx(observations, self.values, self.factors,self.max_encoder_length,self.max_prediction_length)
        quantiles = self.__make_predictions(self.model, data, self.max_prediction_length)   
        anomalies = self.__detect_anomalies(observations[self.values],quantiles)
        return anomalies



if __name__ == "__main__":

    data = pd.read_csv("series.csv")
    detector = TemporalFusionTransformer_AN("ventas","series")

    detector.fit(data)
    print(detector.detect(data))

