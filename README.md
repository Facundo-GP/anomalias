# Detección de Anomalías en Series Esparsas

## Ejecución de la aplicación
```bash
$ cd app/
$ docker build -t app .
$ docker run -p 8000:8000 app
```
Las credenciales de acceso a InfluxDB se encuentran en el archivo `app/config.ini`.

## Ejecución de la aplicación con instancia local de InfluxDB
```bash
$ docker-compose up -d
```
