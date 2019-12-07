# FROM tensorflow/tensorflow:1.14.0-gpu-py3
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM araffin/stable-baselines

#https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ARG DEBIAN_FRONTEND=noninteractive

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y software-properties-common && apt-get update -y
RUN apt-get install --reinstall ca-certificates
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gcc \
    wget \
    unzip \
    libc-dev\
    git \
    bzip2 \
    python3.6 \
    python3-pip \
    python3-setuptools \
    python3-setuptools-git \
    python3.6-dev \
    xvfb \
    ffmpeg \
    ufw \
    freeglut3-dev \
    libgtk2.0-dev \
    libglib2.0-0 \
    libopenmpi-dev \
    zlib1g-dev \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    python3.6-tk && \
    rm -rf /var/lib/apt/lists/*
 
# Install and configure ssh server
# https://docs.docker.com/engine/examples/running_ssh_service/
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server vim nano htop
RUN echo 'PermitRootLogin no\nSubsystem sftp internal-sftp' > /etc/ssh/sshd_config
EXPOSE 22
RUN groupadd sshgroup


# Build and install nvtop
RUN apt-get update && apt-get install -y cmake libncurses5-dev git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /work/*
# DCMAKE_LIBRARY_PATH fix --> see: https://github.com/Syllo/nvtop/issues/1
# RUN cd /tmp && \
#     git clone https://github.com/Syllo/nvtop.git && \
#     mkdir -p nvtop/build && cd nvtop/build && \
#     cmake .. -DCMAKE_LIBRARY_PATH="/usr/local/cuda-9.0/targets/x86_64-linux/lib" && \
#     make && \ 
#     make install && \
#     cd / && \
#     rm -r /tmp/nvtop
    
# Install and configure screen
RUN apt-get update && apt-get install -y --no-install-recommends screen
COPY .screenrc /root/.screenrc


# Expose ports for tensorboard
EXPOSE 7000
EXPOSE 7001

RUN rm -r /workspace; mkdir /workspace
COPY requirements.txt /workspace
RUN python3.6 -m pip install --upgrade pip setuptools wheel
RUN pip3.6 install -r /workspace/requirements.txt
# RUN pip3 install -e git://github.com/duckietown/gym-duckietown.git@aido2#egg=gym-duckietown
RUN git clone https://github.com/duckietown/gym-duckietown.git --branch aido2
COPY maps/* gym-duckietown/gym_duckietown/maps/
RUN pip3.6 install -e gym-duckietown
COPY maps/* /usr/local/lib/python3.6/dist-packages/duckietown_world/data/gd1/maps/

# Clone CornerNet
RUN git clone https://github.com/pezosanta/CornerNet.git
COPY ./BDD100K.sh CornerNet/

RUN mkdir /var/run/sshd
# CMD ["/usr/sbin/sshd", "-D"]
CMD service ssh start && /bin/bash