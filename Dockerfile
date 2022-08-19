FROM pytorchlightning/pytorch_lightning

RUN apt-get update && apt-get upgrade -y && pip3 install --upgrade pip
RUN apt-get install tmux -y && echo "set -g mouse on" > ~/.tmux.conf

RUN pip3 install dvc

WORKDIR /DeepMS
COPY ./requirements.txt /DeepMS/

RUN pip3 install -r requirements.txt