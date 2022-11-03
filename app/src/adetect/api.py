import configparser
import pandas as pd
import nest_asyncio
import uvicorn
from fastapi import FastAPI
from influxdb_client import InfluxDBClient
from pydantic import BaseModel
from typing import List

from . import log

logger = log.logger('API')


class DataFrame(BaseModel):
    index: list
    values: list
    metrics: List[str]
    name: str


class CsvData(BaseModel):
    series: List[str]
    value: List[str]
    time: List[str]
    bucket:  str
    org: str
    measurement_name: str

config = configparser.ConfigParser()
config.read('config.ini')

influx_url = config['influx']['url']
token = config['influx']['token']
org = config['influx']['org']
bucket = config['influx']['bucket']
timeout = config['influx']['timeout']

api_host = config['anomaly_app']['host']
api_port = config['anomaly_app']['port']


class InfluxApi:
    """
    API para el envío de datos a influxdb
    """
    def __init__(self):
        self.__client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
        self.__write_api = self.__client.write_api()

    def write(self, dataFrame, anomalies, data_name):
        for metric in dataFrame:
            df_out = dataFrame[metric].to_frame()
            df_out = df_out.rename(columns={metric: 'anomaly'})
            df_out = df_out[anomalies[metric]]
            df_out["series"] = ["anomalias_" + data_name for i in range(len(df_out))]

            logger.debug('api.py: anomalies to write (metric %s):', metric)
            logger.debug('\n %s', df_out)
            self.__write_api.write(bucket, org, record=df_out,
                                   data_frame_measurement_name=metric, data_frame_tag_columns=["series"])

    def close(self):
        self.__client.close()


def start(detectors):

    api = FastAPI()

    @api.post("/newTS/{id}")
    def newTS(id: str):
        api = InfluxApi()
        detectors.add(len=0, id=id, api=api)

    @api.post("/setAD/{name}/{id}")
    def setAD(name: str, id: str):
        if name == "adtk":
            detectors.adtk_ad(id=id, model_type='MinClusterAD', n_clusters=2)

        elif name == "fm":
            # Agregar factorization machine a detectors
            pass

    @api.post("/start/{id}")
    def start(id: str):
        detectors.start(id=id)

    @api.post("/stop/{id}")
    def stop(id: str):
        detectors.remove(id=id)

    @api.post("/fit/{id}")
    def fit(id: str, data: DataFrame):
        df = pd.DataFrame(list(zip(data.values, data.metrics)),
                          columns=['values', 'metrics'], index=pd.to_datetime(data.index))
        df = df.pivot(columns='metrics', values='values')

        logger.debug('api.py: call to fit(), data:')
        logger.debug('\n %s', df)

        detectors.fit(id, df)

    @api.post("/detect/{id}")
    async def detect(id: str, data: DataFrame):
        try:

            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')

            logger.debug('api.py: call to detect(), data:')
            logger.debug('\n %s', df)

            detectors.append(id, df, data.name)  # Detection
        except Exception as e:
            logger.error('%s', e, exc_info=True)

    @api.post("/influx/write")
    def write_db(data: CsvData):

        data_dict = {"_time": list(map(int, map(float, data.time))), "ventas": list(map(float, data.value)),
                     "series": data.series}
        df = pd.DataFrame.from_dict(data_dict)
        df['_time'] = pd.to_datetime(df._time)
        df = df.set_index('_time')
        print(df)

        client  =  InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
        write_api = client.write_api()
        write_api.write(data.bucket, data.org, record=df,
                        data_frame_measurement_name='ventas_mensuales', data_frame_tag_columns=['series'])

        client.close()
        return {"request": "ok"}

    @api.post("/influx/tasks")
    def send_task(data: CsvData):
        #  TODO: Agregar
        return {"request": "ok"}

    nest_asyncio.apply()
    uvicorn.run(api, port=int(api_port), host=api_host)
