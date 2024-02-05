FROM ubuntu:22.04

LABEL maintainer="Stability AI"
ENV MODEL_NAME="stabilityai/stable-video-diffusion-img2vid-xt"
ENV SERVING_PORT=8080
# ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_TYPE="stabilityai/stable-video-diffusion-img2vid-xt"
ENV SAVE_PATH="/opt/model"

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    python3-pip \
    python3-dev \
    python3-setuptools \
    ffmpeg \
    libsm6 \
    libxext6 \
    gcc \
    && cd /usr/local/bin \
    && pip3 --no-cache-dir install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Update PATH so that the serve programs are found when the container is invoked.
ENV PATH="/opt/:${PATH}"

# copy the code and requirements
COPY requirements.txt /opt/
COPY prepare_huggingface_data.py /opt/

WORKDIR /opt

# install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch
RUN python3 prepare_huggingface_data.py

COPY entrypoint.py /opt/
COPY svd.py /opt/

EXPOSE $SERVING_PORT
ENTRYPOINT ["sh", "-c", "python3 entrypoint.py $SERVING_PORT"]

# Label image
ARG GIT_BRANCH
ARG GIT_VERSION
LABEL org.opencontainers.image.source https://github.com/Stability-AI/marketplace-containers
LABEL org.opencontainers.image.version $GIT_BRANCH
LABEL org.opencontainers.image.revision $GIT_VERSION
LABEL org.opencontainers.image.vendor "Stability AI"