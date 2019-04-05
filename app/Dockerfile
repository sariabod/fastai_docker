FROM nvidia/cuda:10.0-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
ADD . /app
WORKDIR /app

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p ~/miniconda && rm ~/miniconda.sh
ENV PATH=/root/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=true

# Create a Python 3.7 environment
RUN /root/miniconda/bin/conda update -n base -c defaults conda && ~/miniconda/bin/conda clean -ya
RUN /root/miniconda/bin/conda install conda-build && ~/miniconda/bin/conda create -y --name py37 python=3.7 && ~/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# app specific steps
RUN conda install -y -c pytorch \
    pytorch \
    torchvision \
    cudatoolkit=10.0 \
    && conda clean -ya
RUN conda install -y -c fastai \
    fastai \
    && conda clean -ya
RUN conda install -y -c anaconda \
    flask \
    && conda clean -ya 
RUN conda install -y -c anaconda \
    gunicorn \
    && conda clean -ya

EXPOSE 7500
CMD ["gunicorn", "-b", "0.0.0.0:7500", "app"]
