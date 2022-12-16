FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9-cuda11.1.1

# ARG USERNAME=ruggeri
# ARG USER_UID=2565
# ARG USER_GID=2250

# Create the user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -G root -m $USERNAME 
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # && apt-get update \
    # && apt-get install -y sudo \
    # && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    # && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get upgrade -y

WORKDIR /DeepMS
COPY ./requirements.txt .

VOLUME ["/data", "/tokens"]

RUN pip install --upgrade pip && pip install -r requirements.txt


# USER $USERNAME