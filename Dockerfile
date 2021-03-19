FROM continuumio/miniconda3

# Working directory
WORKDIR /lsmutils

# Install apt packages
RUN apt-get update && apt-get install -y unzip
RUN apt-get install -y build-essential cmake
RUN apt-get install -y mpich
RUN apt-get install -y gdal-bin libgdal-dev
RUN apt-get install -y nco

# Download TauDEM
WORKDIR /lsmutils/src
ARG taudem_url=https://github.com/dtarb/TauDEM/archive/v5.3.8.zip
RUN wget --no-check-certificate -O /lsmutils/src/taudem.zip $taudem_url
RUN unzip /lsmutils/src/taudem.zip

# Compile TauDEM
WORKDIR /lsmutils/src/TauDEM-5.3.8/src/build
RUN ls ..
RUN cmake ..
RUN make && make install

# Compile Flowgen
# COPY ./src_rout_prep /code/src_setup
# RUN cd src_setup && gcc -lm -o flowgen flowgen.c

# Set up the anaconda environment
COPY environment.yml /lsmutils
RUN conda env create -f /lsmutils/environment.yml
RUN conda update --all -n lsmutils
RUN conda list -n lsmutils
