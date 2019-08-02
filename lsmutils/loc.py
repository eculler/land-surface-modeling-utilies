import calendar
import copy
import datetime
import logging
from math import floor, ceil
import os
import pandas as pd
import yaml

from .dataset import (
    GDALDataset,
    BoundaryDataset,
    NetCDFDataset,
    DataFrameDataset
)

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
    filename = ''
    dirname = '{temp_dir}'
    default_ext = ''
    variable = ''
    url = ''
    csvargs = {}

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __init__(self, file_id='', filename='', dirname='', default_ext='',
                 variable='', url='', csvargs={}, omit_ext=False,
                 template=None, **env):
        if template:
            self.__init__(**template.info, **template.env)
            return

        self.file_id = file_id if file_id else self.file_id
        ## Is this necessary?
        self.env = env
        self.filename = filename if filename else self.filename
        self.dirname = dirname if dirname else self.dirname
        self.default_ext = default_ext if default_ext else self.default_ext
        self.variable = variable if variable else self.variable
        self.url = url if url else self.url
        self.csvargs = csvargs if csvargs else self.csvargs
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

        # Render filename variables
        self.filename = self.filename.format(**self.env)

        # Render directory name
        self.dirname = dirname if dirname else self.dirname
        self.dirname = self.dirname.format(**self.env)
        if not os.path.isabs(self.dirname):
            self.dirname = os.path.join(self.base_dir, self.dirname)

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
    def info(self):
        return {
            'file_id': self.file_id,
            'filename': self.filename,
            'dirname': self.dirname,
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

        self.id = id if id else self.id
        for id, fmt in self.id.items():
            if not id in filename and not id in dirname:
                filename = '.'.join([filename, fmt])

        self.file_id = file_id
        self.filename = filename
        self.dirname = dirname
        self._default_ext = default_ext
        self.variable = variable
        self.url = url
        self.omit_ext = omit_ext
        self.env = env

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

    def configure(self, cfg, file_id='', dirname=''):
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
            cols.extend(LocatorCollection.__children__()[dim].cols)
        return self.meta.groupby(cols).agg(list)

    def get_subset(self, indices):
        subset = copy.copy(self)
        subset.locs = [self.locs[i] for i in indices]
        subset.meta = self.meta.iloc[indices, :]
        return subset

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
        return '\n    '.join(self.path)


class ListLoc(LocatorCollection):

    yaml_tag = '!DataList'
    loc_type = 'list'
    id = {'': ''}

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
                self.locs.append(Locator(**self.info, **record, **env))

        if records:
            self.meta = pd.DataFrame.from_records(records, index='i')


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
            print(record)
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

        for i, datetime in enumerate(datetimes):
            record = {
                'i': i,
                'datetime': datetime
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
                    collection.meta[key] = value
                new_meta.append(collection.meta)

            self.locs = new_locs
            self.meta = pd.concat(new_meta, ignore_index=True)
