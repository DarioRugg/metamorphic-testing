FROM pytorchlightning/pytorch_lightning

WORKDIR /DeepMS
COPY ./requirements.txt ./DeepMS

RUN apt-get update && apt-get upgrade -y
RUN pip install -r requirements.txt