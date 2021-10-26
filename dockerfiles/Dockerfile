FROM nvidia/cuda:11.1-devel-ubuntu20.04
# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    libopenmpi-dev \
    sudo \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /json_1d_resnet_project
WORKDIR /json_1d_resnet_project

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \ 
    chown -R user:user /json_1d_resnet_project
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh

# packages
RUN conda install pytorch torchvision cudatoolkit=11.1 pytorch-lightning \
    numpy pandas matplotlib scikit-learn tqdm jupyter \
    einops flake8-black flake8 isort loguru -c pytorch -c conda-forge

# install self as module
COPY ./setup.py /json_1d_resnet_project/setup.py
RUN pip install --user --no-cache-dir -e .

# Set the default command to python3
CMD ["python3"]