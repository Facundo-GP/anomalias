#!/bin/bash
docker stop anomalias_influxdb_1
docker stop anomalias_api_1
(cd ./../app/src/ && docker build -t adetect-app:dev .)
docker-compose up -d
