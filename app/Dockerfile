FROM facundogp/anom-api:v2-fastFM

WORKDIR /code

ADD ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

ADD src ./app/src

WORKDIR /code/app/src

CMD ["python", "-m", "adetect"]
