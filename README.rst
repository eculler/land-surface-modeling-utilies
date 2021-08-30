Land Surface Modeling Utilities

This package contains an archive of scripts used to set up land surface models
such as the Distributed Hydrology Soil Vegetation model using free and
open-source software. Each setup step is encapsulated in an Operation object.
Input and output file paths and operation sequences can be specified using YAML
(see example_cfg/niwot.cfg.yaml as an example)

It is recommended to run this program using the supplied Dockerfile. The
docker container uses a conda environment, which must be activated when
running. An example script to build a docker container run this program can be
found below:

```
#/bin/bash
CFGFILE=$1

docker build ~/path/to/lsmutils/code -t lsmutils
docker run --rm -it\
  -v /path/to/data/directory:/example/data \
  -v /path/to/lsmutils/code:/example/src/lsmutils
  -t lsmutils \
  /bin/bash -c \
  "conda run -n lsmutils \
   pip install --ignore-installed -e /example/src/lsmutils && \
   conda run -n lsmutils \
   python -m lsmutils /example/cfg/'$CFGFILE'.cfg.yaml"
```
