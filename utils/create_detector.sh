#!/bin/bash

echo -n "Ingrese id del proceso a crear: "
read id

curl -X 'POST' \
  "http://localhost:8000/newTS/${id}" \
  -H 'accept: application/json' \
  -d '' > /dev/null 2>  /dev/null

sleep 0.2

curl -X 'POST' \
  "http://localhost:8000/setAD/adtk/${id}" \
  -H 'accept: application/json' \
  -d '' > /dev/null 2>  /dev/null

sleep 0.2

curl -X 'POST' \
  "http://localhost:8000/start/${id}" \
  -H 'accept: application/json' \
  -d '' > /dev/null 2>  /dev/null

sleep 0.2
