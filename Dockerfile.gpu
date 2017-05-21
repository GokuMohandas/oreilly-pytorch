FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Goku Mohandas

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libpng-dev \
        pkg-config \
        python \
        python-dev \
        python-tk \
        rsync \
        software-properties-common \
        unzip \
        vim \
        lsof \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        matplotlib \
        numpy \
        scipy \
        Pillow \
        virtualenv

# Step into working directory
WORKDIR /root/oreilly-pytorch

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy all files
COPY . .




