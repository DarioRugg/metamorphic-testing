FROM deepms-torch-image-upgraded

WORKDIR /DeepMS
COPY ./requirements.txt /DeepMS/

RUN pip install -r requirements.txt