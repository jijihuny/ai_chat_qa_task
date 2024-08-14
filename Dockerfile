FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG GITHUB_REPO_ID=https://github.com/jijihuny/ai_chat_qa_task
ARG ENTRYPOINT_FILE_NAME=entrypoint.sh

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y \
    build-essential \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-{dev,distutils}
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN git clone ${GITHUB_REPO_ID} work
RUN cd work
RUN pip install -e .

ENTRYPOINT ${ENTRYPOINT_FILE_NAME}