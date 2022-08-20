#!/bin/bash
docker stop anomalias_influxdb_1
docker stop anomalias_api_1
(cd ./../python_api/ && docker build -t service-api .)
docker-compose up -d
