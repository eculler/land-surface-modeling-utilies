#/bin/bash
ID=$1

docker build ~/Documents/lsmutils -t eculler/land-surface-modeling-utilities
docker run --rm -it\
  --security-opt seccomp=unconfined \
  -v /Volumes/LabShare/Global/SRTMv3:/$ID/data/SRTMv3 \
  -v /Volumes/WD-Data/data/by.project/matilija:/$ID/data \
  -v /Volumes/WD-Data/models/DHSVM/DHSVM3.2-ubuntu/DHSVM/program:/$ID/scripts \
  -v ~/Documents/lsmutils:/$ID/src/lsmutils \
  -v ~/Documents/research/example.setup/cfg:/$ID/cfg \
  -v ~/Documents/research/example.setup/dhsvm:/$ID/dhsvm \
  -v ~/Documents/research/example.setup/tmp:/$ID/tmp \
  -t eculler/land-surface-modeling-utilities \
  /bin/bash -c \
  "conda run -n lsmutils \
   pip install --ignore-installed -e /'$ID'/src/lsmutils && \
   conda run -n lsmutils \
   python -m lsmutils /'$ID'/cfg/'$ID'.cfg.yaml"
