FROM facundogp/anom-api:v2-fastFM

WORKDIR /code

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

RUN pip install --upgrade --user hmmlearn

COPY ./app ./app

WORKDIR /code/app

CMD [ "python", "main.py"] 
