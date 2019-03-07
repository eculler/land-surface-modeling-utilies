from osgeo import gdal
import logging
import os, sys, shutil
import yaml
from multiprocessing import Pool

from calibrate import CaseCollection
from utils import CoordProperty
from operation import LatLonToShapefileOp
from sequence import seqs
from path import TempPath

def run_case(case):
    # Write new files matching case parameters
    case.write_file()

    # Compute Fraction File
    case.dir_structure.datasets.update(seqs.fraction.run(case))



if __name__ == '__main__':
    # Load configuration file
    cfg = yaml.load(open('config/fraction.yaml', 'r').read())
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
    
    run_case(case)
