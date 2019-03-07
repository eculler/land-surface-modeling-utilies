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
    cfg = yaml.load(open('config/forcing.yaml', 'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.info(cfg)

    cfg['case_id']  = 'common'
    cfg['basin_id'] = 'sb'
    collection = CaseCollection(cfg)
    cases = collection.cases
    case = cases[0]
    
    # Clear and create directory for result files
    temp_path = TempPath('', **cfg)
    if os.path.exists(temp_path.dirname):
        shutil.rmtree(temp_path.dirname)
    os.mkdir(temp_path.dirname)


    gages = ['11114495', '11111500']
    for gage_id in gages:
        # Generate case configuration
        cfg['basin_id'] = gage_id
        for raster in os.listdir(cfg['files']['raster'].dirname):
            raster_id = raster.replace('NLDAS_FORB0125_H.A', '')
            cfg['case_id'] = '{}_{}'.format(gage_id, raster_id)
            cfg['files']['raster'].filename = os.path.splitext(raster)[0]
            collection = CaseCollection(cfg)
            cases = collection.cases
            case = cases[0]

            # Clip Raster
            seqs.sum.run(case)
