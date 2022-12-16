FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9-cuda11.1.1

RUN apt-get update && apt-get upgrade -y

WORKDIR /DeepMS
COPY ./requirements.txt .

VOLUME ["/data", "/tokens"]

RUN pip install --upgrade pip && pip install -r requirements.txt
