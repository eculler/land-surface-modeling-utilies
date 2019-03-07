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

def create_routing_file(routing_path, run_config, boundary_ds,
                        coarse_clip_ds, fine_ds, coarse_mask_ds):
    # Skip if routing file already exists
    if os.path.exists(routing_path.nc):
        return
    
    # Flow Direction
    # Flow direction calculated directly at the coarse resolution will be
    # missing edge squares. Calculate at high resolution, then upscale
    no_sinks_ds = RemoveSinksOp(
            run_config,
            input_ds=fine_ds).save()
    
    fine_flow_dir_ds = FlowDirectionOp(
            run_config,
            dem_ds=no_sinks_ds).save()

    fine_flow_acc_ds = FlowAccumulationOp(
            run_config,
            flow_dir_ds=fine_flow_dir_ds).save()

    clip_fine_flow_acc_ds = FineClipOp(
            run_config,
            input_ds=fine_flow_acc_ds,
            boundary_ds=boundary_ds).save()

    coarse_flow_dir_ds = UpscaleFlowDirectionOp(
            run_config,
            flow_acc_ds=clip_fine_flow_acc_ds,
            template_ds=coarse_clip_ds,
            quiver=True).save()
    coarse_flow_dir_ds.saveas('asc')

    taudem_flow_dir_ds = RVICToTauDEMOp(
            run_config,
            input_ds = coarse_flow_dir_ds,
    ).save()

    # Source Area
    source_area_ds = FlowAccumulationOp(
            run_config,
            flow_dir_ds=taudem_flow_dir_ds).save()
    
    # Flow distance
    flow_dist_ds = FlowDistanceOp(
            run_config,
            flow_dir_ds=coarse_flow_dir_ds).save()

    basin_id_ds = BasinIDOp(
            run_config,
            input_ds=coarse_mask_ds).save()


    ## Compile routing as netCDF
    routing_vars = {
        'flow_distance': (flow_dist_ds, 'm'),
        'flow_direction': (coarse_flow_dir_ds, 'd8'),
        'source_area': (source_area_ds, 'grid cells'),
        'basin_id': (basin_id_ds, 'unitless')
    }
    write_netcdf(routing_path.nc, routing_vars)
