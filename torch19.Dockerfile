# FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9-cuda11.1.1

RUN apt-get update && apt-get upgrade -y && pip3 install --upgrade pip
RUN apt-get install tmux -y && echo "set -g mouse on" > ~/.tmux.conf

WORKDIR /DeepMS
COPY ./requirements.txt .

VOLUME [ "/data" ]

RUN pip3 install -r requirements.txt