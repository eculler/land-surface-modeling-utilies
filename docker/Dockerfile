FROM continuumio/anaconda3

# Working directory
WORKDIR /code

# Anaconda doesn't like sh
SHELL [ "/bin/bash", "-c" ]

# Use conda-forge instead of defaults
RUN conda config --remove channels defaults
RUN conda config --add channels conda-forge

# Make the environment
RUN conda create --name hsenv

# Install libraries
RUN conda install -n hsenv -c conda-forge gdal python=3.6
RUN conda install -n hsenv -c conda-forge -c landlab landlab python=3.6
RUN conda install -n hsenv -c conda-forge geos python=3.6
RUN conda install -n hsenv -c conda-forge matplotlib python=3.6
RUN conda install -n hsenv -c conda-forge geopandas python=3.6

# Compile TauDEM from scratch
COPY ./TauDEM /code/TauDEM
RUN apt-get update && apt-get install -y build-essential cmake
RUN apt-get install -y mpich
RUN apt-get install -y gdal-bin libgdal-dev
RUN cd /code/TauDEM/src && mkdir build
RUN cd /code/TauDEM/src/build && cmake ..
RUN cd /code/TauDEM/src/build && make && make install

# Compile Flowgen
COPY ./src_rout_prep /code/src_setup
RUN cd src_setup && gcc -lm -o flowgen flowgen.c

# Copy files over
#COPY ./sb_hotspots_cfg.yaml /code/sb_hotspots_cfg.yaml
COPY ./src /code/src

# Run script on startup
CMD ["/bin/bash", "-c", "source activate hsenv && python -u /code/src/for_Andrew.py"]
