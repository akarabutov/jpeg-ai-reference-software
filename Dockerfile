FROM nvcr.io/nvidia/tensorrt:19.12-py3


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

WORKDIR /root/vm/src/codec/entropy_coding/cpp_exts

RUN \
    source /root/miniconda3/bin/activate && \
    conda activate jpeg_ai_vm && \
    ./build.sh

WORKDIR /root/vm/src/train/3rdparty/apex

RUN \
    source /root/miniconda3/bin/activate && \
    conda activate jpeg_ai_vm && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option=build_ext --global-option="-I/usr/local/cuda/include/" ./

WORKDIR /root/vm

RUN \
    source /root/miniconda3/bin/activate && \
    conda activate jpeg_ai_vm && \
    dvc pull data/test/*.dvc && \
    dvc pull models/VM/*.dvc 
    

RUN echo 'root:Ai123456!@#$%^' | chpasswd

# RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# RUN service ssh start


CMD ["/usr/sbin/sshd","-D"]
