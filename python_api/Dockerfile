FROM python:3.9

WORKDIR /code

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./app ./app

WORKDIR /code/app

CMD [ "python3", "main.py"] 
