FROM nvcr.io/nvidia/tensorrt:19.12-py3


# Install requirements
RUN \
    apt update && \
    apt install openssh-server -y && \
    apt install sudo net-tools -y && \
    apt install ffmpeg libsm6 libxext6 -y && \
    apt install git-lfs -y && \
    service ssh start

WORKDIR /root

# Install python interpreter via Miniforge - conda-forge's own installer
# (BSD-licensed, defaults to the conda-forge channel), instead of Anaconda's
# Miniconda.
RUN \
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \
    chmod +x miniforge.sh && \
    ./miniforge.sh -b -p /root/miniforge3 && \
    rm miniforge.sh

WORKDIR /root/vm

COPY . /root/vm/

# The repo is checked out on Windows with core.autocrlf=true, so shell scripts
# and Makefiles arrive with CRLF line endings, which bash/make can't run.
RUN \
    find scripts src -type f \( -name "*.sh" -o -name "Makefile" \) -exec sed -i 's/\r$//' {} +

RUN \
    source /root/miniforge3/bin/activate && \
    conda init bash && \
    conda create -y -n jpeg_ai_vm python=3.8 && \
    conda activate jpeg_ai_vm && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools==59.5.0 && \
    pip install --no-cache-dir numpy==1.19.1 && \
    pip install --no-cache-dir -r requirements.txt

RUN \
    source /root/miniforge3/bin/activate && \
    conda activate jpeg_ai_vm && \
    pre-commit install

RUN \
    source /root/miniforge3/bin/activate && \
    conda activate jpeg_ai_vm && \
    bash scripts/build_ec_lib.sh jpeg_ai_vm

# NVIDIA Apex is not vendored in this repository (src/train/3rdparty/ does not
# ship - see docs/architecture/13-training.md). Install it separately if you
# need --amp training, or run training with --amp 0.
#
# WORKDIR /root/vm/src/train/3rdparty/apex
#
# RUN \
#     source /root/miniforge3/bin/activate && \
#     conda activate jpeg_ai_vm && \
#     pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option=build_ext --global-option="-I/usr/local/cuda/include/" ./

WORKDIR /root/vm

RUN \
    git lfs install --force && \
    git lfs fetch && \
    git lfs checkout

RUN echo 'root:Ai123456!@#$%^' | chpasswd

# RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# RUN service ssh start


CMD ["/usr/sbin/sshd","-D"]
