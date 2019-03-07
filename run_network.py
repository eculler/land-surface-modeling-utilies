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
    cfg = yaml.load(open('config/dhsvm_setup_niwot.yaml', 'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.debug(cfg)

    # Generate case configurations for 
    collection = CaseCollection(cfg)
    cases = collection.cases
    case = cases[0]

    # Create outlet shapefile from coordinate
    LatLonToShapefileOp(
            cfg, path=case.dir_structure.paths['outlet'],
            coordinate=cfg['info']['outlet']).saveas('shp', working_ext='shp')
    
    # Clear and create directory for result files
    temp_path = TempPath('', **cfg)
    if os.path.exists(temp_path.dirname):
        shutil.rmtree(temp_path.dirname)
    os.mkdir(temp_path.dirname)

    # Write new files matching case parameters
    case.write_file()

    # Compute Network File
    #case.dir_structure.datasets.update(seqs.mask.run(case))
    case.dir_structure.datasets.update(seqs.network.run(case))

    
