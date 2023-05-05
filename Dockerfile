FROM python:3.9-bullseye
ENV PYTHONPATH "/app/"

COPY . /app
WORKDIR /app/active_bayesify/

RUN pip3 install -r ./requirements.txt
WORKDIR /app/active_bayesify/final



