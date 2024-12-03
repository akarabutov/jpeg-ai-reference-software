FROM nvcr.io/nvidia/tensorrt:19.12-py3
ENTRYPOINT /bin/bash


# Install requirements
RUN \
    apt update && \
    apt install openssh-server -y && \
    apt install sudo net-tools -y && \
    apt install ffmpeg libsm6 libxext6 -y && \
    service ssh start

WORKDIR /root

# Install python interpretor
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh  && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    mkdir vm

COPY . /root/vm/

WORKDIR /root/vm

RUN \
    source /root/miniconda3/bin/activate && \
    conda init bash && \
    conda env create -f environment.yml

WORKDIR /root/vm

RUN \
    source /root/miniconda3/bin/activate && \
    conda activate jpeg_ai_vm && \
    pre-commit install 

RUN echo 'root:Ai123456!@#$%^' | chpasswd

# RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# RUN service ssh start

RUN source /root/miniconda3/bin/activate && conda activate jpeg_ai_vm && /bin/bash


CMD ["/usr/sbin/sshd","-D"]
