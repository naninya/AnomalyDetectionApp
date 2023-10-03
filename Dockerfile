# FROM python:3.9

# WORKDIR /app
# COPY ./src /src
# COPY ./requirements.txt .


# # COPY ./requirements.txt .
 
# # RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

#  RUN apt-get update && \
#     # Install system dependencies
#     apt-get install -y --no-install-recommends libaio1 gcc && \
#     pip install --upgrade pip && \
#     pip install -r ./requirements.txt && \
#     # Cleanup
#     apt-get autoremove -y && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*


ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get install wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN conda install -c pytorch faiss-gpu -y
RUN conda clean --all


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /app/src


# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

