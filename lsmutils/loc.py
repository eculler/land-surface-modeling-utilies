import calendar
import copy
import datetime
import glob
import logging
from math import floor, ceil
import numpy as np
import os
import pandas as pd
import re
import yaml

from .dataset import (
    GDALDataset,
    BoundaryDataset,
    NetCDFDataset,
    DataFrameDataset
)

def merge_lists(values):
    merged = []
    for value in values:
        try:
            merged.extend(value)
        except TypeError:
            merged.append(value)
    return merged

class PathSegments(yaml.YAMLObject):

    yaml_tag = u"!PathSegments"

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return os.path.join(*fields['segments'])


class LocatorMeta(yaml.YAMLObjectMetaclass):

    def __children__(cls):
        children = {cls.loc_type: cls}
        for child in cls.__subclasses__():
            if child.loc_type:
                children.update({child.loc_type: child})
            children.update(child.__children__())
        return children

class Locator(yaml.YAMLObject, metaclass=LocatorMeta):
    """
    A class to keep track of file structures
    """

    yaml_tag = '!File'
    loc_type = 'file'
    file_id = ''

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __init__(self,
                 file_id='', filename='', dirname='{temp_dir}', default_ext='',
                 variable='', url='', csvargs={}, omit_ext=False,
                 template=None, **env):
        if template:
            self.__init__(**template.info, **template.env)
            return

        self.file_id = file_id if file_id else self.file_id
        self.env = env
        self.base_dir = ''
        self._filename_fmt = filename
        self._filename = ''
        self._dirname_fmt = dirname
        self._dirname = ''
        self.default_ext = default_ext
        self.variable = variable
        self.url = url
        self.csvargs = csvargs
        self.omit_ext = omit_ext

        self._dataset = None

    def configure(self, cfg, file_id='', dirname=''):
        self.file_id = file_id if file_id else self.file_id
        self.env.update(cfg)

        # Set base directory
        if 'base_dir' in cfg:
            self.base_dir = cfg['base_dir']
        else:
            self.base_dir = os.getcwd()

        # Render url
        try:
            self.url = self.url.format(**self.env)
        except:
            pass

        # Create directory if it does not exist
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        return self

    @property
    def filename(self):
        if not self._filename:
            self.filename = self._filename_fmt
        return self._filename

    @filename.setter
    def filename(self, new_filename):
        try:
            self._filename = new_filename.format(**self.env)
        except KeyError:
            self._filename = new_filename
            self._filename_fmt = new_filename
        return self._filename

    @property
    def dirname(self):
        if not self._dirname:
            self.dirname = self._dirname_fmt
        return self._dirname

    @dirname.setter
    def dirname(self, new_dirname):
        try:
            self._dirname = new_dirname.format(**self.env)
            if not os.path.isabs(self._dirname):
                self._dirname = os.path.join(self.base_dir, self._dirname)

        except KeyError:
            self._dirname = ''
            self._dirname_fmt = new_dirname

        return self._dirname

    @property
    def info(self):
        return {
            'file_id': self.file_id,
            'filename': self._filename,
            'dirname': self._dirname,
            'default_ext': self.default_ext,
            'variable': self.variable,
            'url': self.url,
            'omit_ext': self.omit_ext
        }

    def _path(self, filename, extension):
        if extension:
            filename = '.'.join([filename, extension])
        return os.path.join(self.dirname, filename)

    def ext(self, extension):
        return self._path(self.filename, extension)

    @property
    def path(self):
        if self.omit_ext:
            return self.no_ext
        if self.default_ext:
            return self.ext(self.default_ext)
        return self.no_ext

    @property
    def exists(self):
        return os.path.exists(self.path)

    @property
    def isfile(self):
        return os.path.isfile(self.path)

    @property
    def cfg(self):
        return self.ext('cfg')

    @property
    def gtif(self):
        return self.ext('gtif')

    @property
    def tif(self):
        return self.ext('tif')

    @property
    def nc(self):
        return self.ext('nc')

    @property
    def png(self):
        return self.ext('png')

    @property
    def asc(self):
        return self.ext('asc')

    @property
    def shp(self):
        return self.ext('shp')

    @property
    def txt(self):
        return self.ext('txt')

    @property
    def csv(self):
        return self.ext('csv')

    @property
    def dat(self):
        return self.ext('dat')

    @property
    def no_ext(self):
        return self.ext(None)

    @property
    def dataset(self):
        if not self._dataset:
            self._dataset = self.get_dataset(self)
        return self._dataset

    def get_dataset(self, loc):

        filetype = self.default_ext if self.default_ext else ''

        if filetype=='shp':
            try:
                return BoundaryDataset(loc)
            except Exception:
                logging.error(exc)
                logging.error('{} failed to load as shapefile'.format(
                    path.path))

        if filetype in DataFrameDataset.filetypes:
            return DataFrameDataset(loc, filetype=filetype, **self.csvargs)

        if filetype in NetCDFDataset.filetypes:
            return NetCDFDataset(loc)

        if filetype in GDALDataset.filetypes:
            return GDALDataset(loc, filetype=filetype)

        return None

    def __str__(self):
        return self.path


class LocatorCollection(Locator):

    loc_type = None
    cols = NotImplemented
    id = NotImplemented

    def __init__(self, file_id='', filename='',
                 dirname='', default_ext='', variable='', url='',
                 omit_ext=False, template=None, id={}, **env):
        if template:
            self.__init__(**template.info, **template.env, **env)
            return
        self.env = env
        self.base_dir = ''

        self.id = id if id else self.id
        for id, fmt in self.id.items():
            if not id in filename and not id in dirname:
                filename = '.'.join([filename, fmt])

        self.file_id = file_id if file_id else self.file_id
        self.env = env
        self.base_dir = ''
        self._filename_fmt = filename
        self._filename = ''
        self._dirname_fmt = dirname
        self._dirname = ''
        self._default_ext = default_ext
        self.variable = variable
        self.url = url
        self.omit_ext = omit_ext

        self.i = 0
        self.meta = pd.DataFrame()
        self.locs = []

        try:
            self.expand(**env)
        except TypeError:
            logging.warning('Required locator metadata missing: %s', self.id)

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def configure(self, cfg, file_id='', dirname='', dims=[]):
        self.file_id = file_id if file_id else self.file_id
        self.dirname = dirname if dirname else self.dirname
        for loc in self.locs:
            loc.configure(cfg, file_id, dirname)
        self.meta[self.file_id] = self.meta.index
        return self

    @property
    def isfile(self):
        return any([loc.isfile for loc in self.locs])

    @property
    def dataset(self):
        ds_list = [loc.dataset for loc in self.locs]
        if len(ds_list) == 1:
            return ds_list[0]
        meta_iter = self.meta.to_dict('records')
        for ds in ds_list:
            if hasattr(ds, 'meta'):
                ds.meta = meta_iter.pop(0)
        return ds_list

    @property
    def path(self):
        pth_list = [loc.path for loc in self.locs]
        if len(pth_list) == 1:
            return pth_list.pop()
        return pth_list

    def ext(self, extension):
        pth_list = [loc.ext(extension) for loc in self.locs]
        if len(pth_list) == 1:
            return pth_list.pop()
        return pth_list

    @Locator.filename.setter
    def filename(self, new_filename):
        for loc in self.locs:
            loc.filename = new_filename
        self._filename = new_filename

    @Locator.dirname.setter
    def dirname(self, new_dirname):
        for loc in self.locs:
            loc.dirname = new_dirname
        self._dirname = new_dirname

    @property
    def default_ext(self):
        return self._default_ext

    @default_ext.setter
    def default_ext(self, new_ext):
        for loc in self.locs:
            loc.default_ext = new_ext
        self._default_ext = new_ext

    def get_dim_values(self, dims):
        # Switch locator type to column names
        cols = []
        for dim in dims:
            # Allow columns configured by the user
            try:
                cols.extend(LocatorCollection.__children__()[dim].cols)
            except KeyError:
                cols.append(dim)
        return self.meta.reset_index().groupby(cols).agg(merge_lists)

    def get_subset(self, indices):
        subset = copy.copy(self)
        subset.locs = [self.locs[i] for i in indices]
        subset.meta = self.meta.iloc[indices, :]
        return subset

    def reduce(self, dims):
        new_locs = []
        dim_values = self.get_dim_values(dims)
        dim_values['i'] = np.arange(len(dim_values))
        for row in dim_values.to_dict('records'):
            new_locs.append([self.locs[i] for i in row[self.file_id]][0])
            new_locs[-1].env['i'] = row['i']

        # Adjust indices
        self.meta = dim_values.reset_index()
        self.meta[self.file_id] = self.meta.i
        self.locs = new_locs

    def get_loc(self, **meta):
        qry = ' and '.join(['{} <= {}'.format(k,v) for k,v in meta.items()])
        return self.locs[self.meta.query(qry)[[self.file_id]]]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.locs):
            raise StopIteration
        loc = self.locs[self.i]
        meta = self.meta.iloc[self.i].to_dict()
        self.i += 1
        return loc, meta

    def __str__(self):
        pth_list = [loc.path for loc in self.locs]
        if len(pth_list) == 1:
            return self.path
        if len(pth_list) > 10:
            return '\n    '.join(self.path[:4] + ['...'] + self.path[-4:])
        return '\n    '.join(self.path)


class ListLoc(LocatorCollection):

    yaml_tag = '!DataList'
    loc_type = 'list'
    cols = ['i']
    id = {'i': '{i}'}

    def expand(self, meta=[], locs=[], **env):
        self.locs = locs.copy()
        records = []

        for i, info in enumerate(meta):
            record = {'i': i}
            record.update(info)
            records.append(record)
            if locs:
                self.locs.append(
                    Locator(template=locs[i], **self.info, **record,
                            **env))
            else:
                record.update(env)
                self.locs.append(Locator(**self.info, **record))

        if records:
            self.meta = pd.DataFrame.from_records(records, index='i')
            self.cols = self.meta.columns.tolist()
        else:
            self.meta = pd.DataFrame()

class MonthlyLoc(LocatorCollection):

    yaml_tag = '!MonthlyFile'
    loc_type = 'monthly'
    cols = ['month', 'month_name', 'month_abbr']
    id = {'month': '{month:02d}'}

    def expand(self, months=range(1,13), **env):
        self.locs = []
        records = []

        for i, month in enumerate(months):
            record = {
                'i': i,
                'month': month,
                'month_name': calendar.month_name[month],
                'month_abbr': calendar.month_abbr[month]
            }
            records.append(record)
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')

class YearlyLoc(LocatorCollection):

    yaml_tag = '!Yearly'
    loc_type = 'yearly'
    cols = ['year']
    id = {'year': '{year}'}

    def expand(self, start, end, **env):
        self.locs = []
        records = []

        for i, year in enumerate(range(start, end+1)):
            record = {
                'i': i,
                'year': year
            }
            records.append(record)
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')


class YearMonthlyLoc(LocatorCollection):

    yaml_tag = '!YearMonthly'
    loc_type = 'yearmonthly'
    cols = ['year', 'month', 'month_name', 'month_abbr', 'month_yday']
    id = {'year': '{year}', 'month': '{month:02d}'}

    def expand(self, start, end, **env):
        self.locs = []
        records = []

        for i, dt in enumerate(pd.date_range(start, end, freq='MS')):
            record = {
                'i': i,
                'year': dt.year,
                'month': dt.month,
                'month_name': calendar.month_name[dt.month],
                'month_abbr': calendar.month_abbr[dt.month],
                'month_yday': dt.timetuple().tm_yday
            }
            records.append(record)
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')


class DatetimeLoc(LocatorCollection):

    yaml_tag = '!DatetimeFile'
    loc_type = 'datetimes'
    cols = ['datetime']
    id = {'datetime': '{datetime:%Y%m%d%H%M%S}'}

    def expand(self, datetimes, **env):
        self.locs = []
        records = []

        for i, dt in enumerate(datetimes):
            record = {
                'i': i,
                'datetime': dt
            }
            records.append(record)
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')

class TileLoc(LocatorCollection):

    yaml_tag = '!Tiles'
    loc_type = 'tiles'
    cols = [
        'min_lon', 'min_lat', 'max_lon', 'max_lat',
        'cardinal_lon', 'cardinal_lat', 'abs_min_lon', 'abs_min_lat']
    id = {
        'lon': '{min_lon}',
        'lat': '{min_lat}'
    }

    def expand(self, bbox=None, **env):

        self.locs = []
        records = []

        min_lon = floor(bbox.min.lon)
        max_lon = ceil(bbox.max.lon)
        min_lat = floor(bbox.min.lat)
        max_lat = ceil(bbox.max.lat)

        for i, (lon, lat) in enumerate(zip(
                range(int(min_lon), int(max_lon)),
                range(int(min_lat), int(max_lat)))):
            record = {
                'i': i,
                'min_lon': min_lon,
                'min_lat': min_lat,
                'max_lon': min_lon + 1,
                'max_lat': min_lat + 1,
                'cardinal_lon': 'E' if min_lon >= 0 else 'W',
                'cardinal_lat': 'N' if min_lat >= 0 else 'S',
                'abs_min_lon': abs(min_lon),
                'abs_min_lat': abs(min_lat)
            }
            records.append(record.copy())
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')

class LayeredLoc(LocatorCollection):

    yaml_tag = '!Layers'
    loc_type = 'layers'
    cols = ['layer_min', 'layer_max']
    id = {'layer': '{layer_min}'}

    def expand(self, layers=[], **env):
        self.locs = []
        records = []
        for i, (min, max) in enumerate(zip(layers[:-1], layers[1:])):
            record = {
                'i': i,
                'layer_min': min,
                'layer_max': max
            }
            records.append(record)
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')


class MultipleTypeLoc(LocatorCollection):

    yaml_tag = '!MultiType'
    loc_type = 'types'
    cols = ['type']
    id = {'type': '{type}'}

    def expand(self, types, **env):
        self.locs = []
        records = []
        for i, type in enumerate(types):
            record = {
                'i': i,
                'type': type
            }
            records.append(record)
            self.locs.append(Locator(**self.info, **record, **env))
        self.meta = pd.DataFrame.from_records(records, index='i')

class RegexLocatorCollection(LocatorCollection):

    yaml_tag = '!Regex'
    loc_type = 'regex'
    id = {'': ''}

    def expand(self, dimensions, filename_re, **env):
        regex = re.compile(filename_re)
        records = []
        files = [os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(self.dirname + '/*')]

        if not files:
            raise ValueError('No files in directory {}'.format(self.dirname))

        for i, file in enumerate(files):
            search = regex.search(file)
            if search:
                record = {'i': i}
                record.update(search.groupdict())
                records.append(record)

        if not records:
            raise ValueError('No files match regex {}'.format(filename_re))

        self.meta = pd.DataFrame.from_records(records, index='i')
        grouped = self.meta.groupby(*dimensions)
        self.locs = []
        dim_meta = []
        for name, group in grouped:
            meta_dict = group.reset_index().to_dict('records')
            listloc = ListLoc(**self.info, meta=meta_dict)
            self.locs.extend(listloc.locs)
            dim_meta.append(listloc.meta)

        self.meta = pd.concat(dim_meta, ignore_index=True)
        self.filename


class ComboLocatorCollection(LocatorCollection):

    yaml_tag = '!Combo'
    loc_type = 'combo'
    id = {'': ''}

    def expand(self, dimensions, **env):
        self.cols = []
        self.locs = [None]
        self.meta = None

        while dimensions:

            new_locs = []
            new_meta = []
            loc_cls = Locator.__children__()[dimensions.pop(0)]


            records = (
                [{}] if (self.meta is None)
                else self.meta.to_dict('records')
            )
            for loc, rec in zip(self.locs, records):
                env.update(rec)
                collection = loc_cls(**self.info, **env)
                self.filename = collection.filename
                self.cols.extend(collection.cols)
                new_locs.extend(collection.locs)

                # Convert dataframe to list of tuples
                for key, value in rec.items():
                    try:
                        collection.meta[key] = value
                    except ValueError:
                        continue
                new_meta.append(collection.meta)

            self.locs = new_locs
            self.meta = pd.concat(new_meta, ignore_index=True)
