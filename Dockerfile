FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get upgrade -y && pip3 install --upgrade pip
RUN apt-get install tmux -y && echo "set -g mouse on" > ~/.tmux.conf

WORKDIR /DeepMS
COPY ./requirements.txt .

VOLUME [ "/data" ]

RUN pip3 install -r requirements.txt