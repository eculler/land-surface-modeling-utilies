import logging
import subprocess
import datetime
import os, shutil
import re

from path import RVICInputTempPath as TempPath
from dataset import GDALDataset
from flux2nc_flow import flux2nc

def run_vic(cfg, dirs, boundary_ds):
    if not os.path.isfile(dirs.files['rvic-forcing']):
        subprocess.call([os.path.join(cfg['base_dir'], cfg['vic_exec']),
                         '-g',
                         dirs.files['vic-global']])

        # VIC output to netCDF for RVIC
        flux2nc(dirs.files['vic-result'],
                dirs.files['rvic-forcing'],
                cfg['vic_fields']
        )

        # Plot forcing data
        if 'vic' in cfg['plot']:
            vic_path = dirs.paths['rvic-forcing']
            vic_ds = GDALDataset(vic_path)
            vic_plot_uri = TempPath('vic', **cfg)
            vic_ds.plot(boundary_ds, vic_plot_uri.png)
