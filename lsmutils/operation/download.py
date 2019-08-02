import logging
import pandas as pd
import requests
import shutil
import subprocess

from lsmutils.loc import *
from lsmutils.operation import Operation, OutputType

class DownloadOpenDASOp(Operation):

    title = 'Download OpenDAS data using ncks'
    name = 'download-opendas'
    output_types = [
        OutputType('chunks', 'nc'),
        OutputType('data', 'nc')
    ]

    def run(self, ds, vars, bbox, start, end, chunk=None):
        # Split large data sets into chunks
        if chunk:
            dates = pd.date_range(start, end, freq=chunk)
        else:
            dates = [start, end]
        self.locs['chunks'] = DatetimeLoc(
            datetimes=dates,
            template=self.locs['chunks'])
        self.locs['chunks'].configure(self.cfg)

        # Download
        for i, (start_date, end_date) in enumerate(zip(dates[:-1], dates[1:])):
            info = {'year': start_date.year}
            download_args = [
                'ncks', '--mk_rec_dmn', 'time',
                '-v', ','.join(vars),
                '-d', ','.join(['time',
                                start_date.isoformat(),
                                end_date.isoformat()]),
                '-d', ','.join(['lon', str(bbox.min.lon), str(bbox.max.lon)]),
                '-d', ','.join(['lat', str(bbox.min.lat), str(bbox.max.lat)]),
                ds.loc.url.format(**info),
                self.locs['chunks'].locs[i].path
            ]
            logging.info('Calling process %s', ' '.join(download_args))
            download_process = subprocess.Popen(
                download_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            download_output, _ = download_process.communicate()

        # Merge chunks
        cat_args = [
            'SKIP_SAME_TIME=1', 'cdo', 'mergetime',
            os.path.join(self.locs['chunks'].dirname, '*'),
            self.locs['data'].path
        ]
        logging.info('Calling process %s', ' '.join(cat_args))
        cat_process = subprocess.Popen(
            cat_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        cat_output, _ = cat_process.communicate()


class DownloadThreddsOp(Operation):

    title = 'Download Thredds data'
    name = 'download-thredds'
    output_types = [
        OutputType('chunks', 'nc'),
        OutputType('data', 'nc')
    ]

    def run(self, ds, vars, bbox, start, end, chunk=None):
        # Split large data sets into chunks
        if chunk:
            dates = pd.date_range(start, end, freq=chunk)
        else:
            dates = [start, end]
        self.locs['chunks'] = DatetimeLoc(
            datetimes=dates,
            template=self.locs['chunks'])
        self.locs['chunks'].configure(self.cfg)

        print(dates)
        # Download
        for i, (start_date, end_date) in enumerate(zip(dates[:-1], dates[1:])):
            info = {'year': start_date.year}

            params = {
                'var': vars,
                'north': bbox.urc.lat,
                'west': bbox.llc.lon,
                'east': bbox.urc.lon,
                'south': bbox.llc.lat,
                'horizStride': 1,
                'time_start': start_date,
                'time_end': end_date,
                'timeStride': 1,
                'accept': 'netcdf'
            }
            r = requests.get(ds.loc.url.format(**info),
                             params=params, stream=True)
            print(self.locs['chunks'].locs[i].path)
            with open(self.locs['chunks'].locs[i].path, 'wb') as file:
                for chunk in r.iter_content(chunk_size=128):
                    file.write(chunk)

            print(r.url)

        # Merge chunks
        cat_args = [
            'SKIP_SAME_TIME=1', 'cdo', 'mergetime',
            os.path.join(self.locs['chunks'].dirname, '*'),
            self.locs['data'].path
        ]
        logging.info('Calling process %s', ' '.join(cat_args))
        cat_process = subprocess.Popen(
            cat_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        cat_output, _ = cat_process.communicate()


class DownloadPolarisOp(Operation):
    """ Download POLARIS data """

    title = 'Download POLARIS data'
    name = 'download-polaris'
    output_types = [
        OutputType('polaris', 'tif')
    ]

    def run(self, input_ds):
        if not input_ds.loc.exists:
            req = requests.get(input_ds.loc.url, stream=True)
            with open(input_ds.loc.path, 'wb') as file:
                shutil.copyfileobj(req.raw, file)
