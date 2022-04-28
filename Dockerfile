FROM continuumio/miniconda3
WORKDIR /workdir
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate env1" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda env create -f environment.yml