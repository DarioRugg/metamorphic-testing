FROM pytorchlightning/pytorch_lightning

RUN apt-get update && apt-get upgrade -y

WORKDIR /DeepMS
COPY ./requirements.txt /DeepMS/

RUN pip install -r requirements.txt