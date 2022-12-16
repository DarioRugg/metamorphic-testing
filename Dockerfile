FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9-cuda11.1.1

RUN apt-get update && apt-get upgrade -y

WORKDIR /DeepMS
COPY ./requirements.txt .

VOLUME ["/data", "/tokens"]

RUN pip install --upgrade pip && pip install -r requirements.txt

# Create the copy of the ruggery user and group
RUN groupadd --gid 2250 \
    && useradd --uid 2565 --gid 2250 -m ruggeri 
