FROM continuumio/miniconda3

RUN mkdir -p ia_coursework

COPY . /ia_coursework

WORKDIR /ia_coursework

RUN conda env update -f environment.yml --name IACW

RUN apt-get update && apt-get install -y \
    git

RUN echo "conda activate IACW" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

RUN git init
RUN pip install pre-commit
RUN pre-commit install
