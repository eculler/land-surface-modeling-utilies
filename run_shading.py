import logging
import yaml
import csv
from osgeo import ogr, osr
import shutil, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

from calibrate import CaseCollection
from utils import CoordProperty
from operation import LatLonToShapefileOp
from sequence import seqs
from path import TempPath

if __name__ == '__main__':
    # Load configuration file
    cfg = yaml.load(open('../landslide/boulder/boulder_cfg.yaml',
                         'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.debug(cfg)

    # Generate case configurations for 
    collection = CaseCollection(cfg)
    cases = collection.cases
    case = cases[0]
    
    # Clear and create directory for result files
    temp_path = TempPath('', **cfg)
    if os.path.exists(temp_path.dirname):
        shutil.rmtree(temp_path.dirname)
    os.mkdir(temp_path.dirname)

    # Compute Network File
    seqs.shading.run(case)
