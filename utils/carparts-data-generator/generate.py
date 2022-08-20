import pandas as pd
import numpy as np
import random
import plotly.express as px
import argparse
import requests
import time
from time import sleep


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


class carpartsDataGenerator():
    def __init__(self, path):
        self.path = path

    def __readDataset(self):
        self.data = pd.read_csv(self.path).fillna(0).to_numpy()

    def __generateData(self,serie_index):
        for j in range(self.lenght):
            i_rand = random.randint(0,self.data.shape[0]-1)
            j_rand = random.randint(0,self.data.shape[1]-1)
            
            self.generated_dataset[serie_index,j] = int(self.data[i_rand,j_rand] / 1e6)

    def __addNoise(self):
        
        for i in range(self.n_series):
            for j in range(self.lenght):
                self.generated_dataset[i,j] += int(np.random.normal(0, self.noise_factor*(self.generated_dataset[i,j] + 8)/3, 1)[0])
                if self.generated_dataset[i,j] < 0.0:
                    self.generated_dataset[i,j] = 0.0

    def generateDataset(self, n_series, lenght, noise_factor):
        
        self.n_series = n_series
        self.lenght = lenght
        self.noise_factor = noise_factor

        self.__readDataset()
        self.generated_dataset = np.zeros((n_series,lenght))
        for i in range(n_series):
            self.__generateData(i)

        self.__addNoise()


    def writeDataset(self,bucket,org,measure_name):
        
        data_dict = {}
        data_dict["series"] = []
        data_dict["value"] = []
        data_dict["time"] = []
        series_list = ["serie{}".format(i) for i in range(self.n_series)]


        for i in range(self.lenght):
            for j in range(self.n_series):
                data_dict["series"].append(series_list[j])
                data_dict["value"].append(str(self.generated_dataset[j,i]))
                sleep(0.00001)
                data_dict["time"].append(str(time.time()*1e9))

        data_dict["bucket"] = bucket
        data_dict["org"] = org
        data_dict["measurement_name"] = measure_name

        url = "http://localhost:8000/influx/write"
        response = requests.post(url,json=data_dict)
        print(response.content)
        
        




if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_series", help="Cantidad de series a generar",type=int,default=10,choices=range(1,101))
    parser.add_argument("--lenght", help="Tamaño de las series a generar",type=int,default=1000,choices=range(1,60001))
    parser.add_argument("--noise_factor", help="Tamaño de las series a generar",type=float,default=0.0,choices=[Range(0.0, 1.0)])
    parser.add_argument("--plot_list", help="Tamaño de las series a generar",type=int,nargs="+",default=[])
    parser.add_argument("--write_influx", help="Escribe datos en influx",type=int,default=0,choices=[0,1])
    parser.add_argument("--bucket", help="Bucket de influx donde escribir los datos",type=str,default="carparts")
    parser.add_argument("--org", help="Organizacion de influx donde escribir los datos",type=str,default="fing")
    parser.add_argument("--measure_name", help="Nombre de la metrica",type=str,default="ventas_mensuales")
    args = parser.parse_args()


    dataset_path = "./carparts.csv"
    generator = carpartsDataGenerator(dataset_path)
    generator.generateDataset(args.n_series,args.lenght,args.noise_factor)
    if args.write_influx:
        generator.writeDataset(args.bucket,args.org,args.measure_name)



