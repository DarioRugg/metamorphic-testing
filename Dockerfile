FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get upgrade -y && pip install --upgrade pip

WORKDIR /DeepMS
COPY ./requirements.txt .

VOLUME ["/data", "/tokens"]

RUN pip install -r requirements.txt