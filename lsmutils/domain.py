# Import GDAL
from osgeo import gdal, ogr, osr
import os, shutil, sys

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import describe as sp_describe
from scipy.io import netcdf
import mpl_toolkits.basemap as bm
import logging
import subprocess
import datetime

from utils import *
from operation import *
        
def create_domain_file(domain_path, run_config, boundary_ds, coarse_ds,
                       fine_ds, coarse_mask_ds):
    # Skip if domain file already exists
    if os.path.exists(domain_path.nc):
        return

    # Coarse
    coarse_clip_ds = CoarseClipOp(
            run_config,
            input_ds=coarse_ds,
            boundary_ds=boundary_ds
    ).saveas('gtif')
    
    # Fine
    fine_mask_ds = FineMaskOp(
            run_config,
            input_ds = fine_ds,
            boundary_ds = boundary_ds).saveas('gtif')
    fine_mask_ds.saveas('asc')
    fine_mask_ds.saveas('nc')

    fine_clip_ds = FineClipOp(
            run_config,
            input_ds=fine_ds,
            boundary_ds=boundary_ds
    ).saveas('gtif')
    
    unique_ds = UniqueOp(run_config, input_ds=coarse_ds).saveas('gtif')
    zones_ds = ZonesOp(
            run_config,
            unique_ds=unique_ds,
            fine_ds=fine_ds
    ).saveas('gtif')


    fraction_ds = FractionOp(
            run_config,
            unique_ds=unique_ds,
            zones_ds=zones_ds,
            fine_mask_ds=fine_mask_ds
    ).saveas('gtif')
    fraction_ds.saveas('nc')

    gridarea_ds = GridAreaOp(
            run_config,
            fraction_ds=fraction_ds
    ).saveas('nc')
    
    # Compile Domain File as netCDF
    domain_vars = {
        'frac': (fraction_ds, 'unitless'),
        'area': (gridarea_ds, 'm2'),
        'mask': (coarse_mask_ds, 'unitless')
    }
    write_netcdf(domain_path.nc, domain_vars)
