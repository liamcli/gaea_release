FROM nvidia/cuda:10.1-cudnn7-runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Setup Ubuntu
RUN apt-get update --yes
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim 

# Get Miniconda and make it the main Python interpreter
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n pytorch_env python=3.6
RUN echo "source activate pytorch_env" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch_env
RUN conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
RUN conda install boto3
RUN pip install scipy
RUN pip install hydra-core
RUN pip install tensorboard
RUN pip install xxhash cachetools

#ARG NAS_VER=unknown
# Always rebuild from here
RUN git clone https://github.com/liamcli/AutoDL-Projects /code/AutoDL
ENV PYTHONPATH /code/AutoDL
RUN mkdir /results

RUN mkdir -p /code/nas-theory
ADD . /code/nas-theory/


COPY scripts/run_hydra_config.sh .
COPY scripts/evaluate_arch.sh .
