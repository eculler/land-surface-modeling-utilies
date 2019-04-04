import calendar
import copy
import datetime
import logging
import os
import pandas as pd
import yaml

from .dataset import GDALDataset, BoundaryDataset, DataFrameDataset

class LocatorMeta(yaml.YAMLObjectMetaclass):

    def __children__(cls):
        children = {cls.loc_type: cls}
        for child in cls.__subclasses__():
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
    csvargs = {}

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __init__(self, file_id='', filename='', dirname='', default_ext='',
                 variable='', csvargs={}, omit_ext=False, **env):
        self.file_id = file_id if file_id else self.file_id
        self.env = env
        self.filename = filename if filename else self.filename
        self.dirname = dirname if dirname else self.dirname
        self.default_ext = default_ext if default_ext else self.default_ext
        self.variable = variable if variable else self.variable
        self.csvargs = csvargs if csvargs else self.csvargs
        self.omit_ext = omit_ext

        self._dataset = None

    def configure(self, cfg, file_id='', dirname=''):
        self.file_id = file_id if file_id else self.file_id
        self.env.update(cfg)
        self.env['file_id'] = self.file_id

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

        # Create directory if it does not exist
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        return self

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

    def get_dataset(self, path):
        if not path.exists:
            return None

        filetype = self.default_ext if self.default_ext else ''

        if filetype=='shp':
            try:
                dataset = BoundaryDataset(path)
                if not dataset.dataset is None:
                    return dataset
                logging.warning('{} failed to load as shapefile'.format(
                    path.path))
            except Exception:
                logging.error(exc)
                logging.error('{} failed to load as shapefile'.format(
                    path.path))

        if filetype in DataFrameDataset.filetypes:
            dataset = DataFrameDataset(path, filetype=filetype, **self.csvargs)
            return dataset

        try:
            dataset = GDALDataset(path, filetype=filetype)
            if not dataset.dataset is None:
                return dataset
            logging.warning('{} failed to load as raster'.format(path.path))
        except Exception as exc:
            logging.error(exc)
            logging.error('{} failed to load as raster'.format(path.path))

        return None


class LocatorCollection(Locator):

    loc_type = 'list'
    cols = ['i']

    def __init__(self, locs, meta=[], file_id='',
                 dirname='', default_ext='', variable='', omit_ext=False,
                 **env):
        self.file_id = file_id
        self.locs = locs
        if not meta:
            for i in range(len(locs) - 1):
                meta_dict = {'i': i}
                meta.append(meta_dict.copy())
                meta_dict.update(env)
        self.meta = pd.DataFrame.from_records(meta)

        self.dirname = dirname
        self._default_ext = default_ext
        self.variable = variable
        self.omit_ext = omit_ext

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

class MonthlyLoc(LocatorCollection):

    yaml_tag = '!MonthlyFile'
    loc_type = 'monthly'
    cols = ['month', 'month_name', 'month_abbr']

    def __init__(self, filename, months=range(1,13),
                 file_id='', dirname='', default_ext='',
                 variable='', omit_ext=False, **env):
        locs = []
        meta = []
        i = 0

        if not 'month' in filename:
            filename += '.{month:02d}'
        for month in months:
            meta_dict = {
                'month': month,
                'month_name': calendar.month_name[month],
                'month_abbr': calendar.month_abbr[month]
            }
            meta.append(meta_dict.copy())
            meta_dict.update(env)
            loc = Locator(
                file_id=file_id,
                filename=filename, dirname=dirname, default_ext=default_ext,
                variable=variable, omit_ext=omit_ext,
                **meta_dict)
            locs.append(loc)
            i += 1

        super(MonthlyLoc, self).__init__(locs, meta, **env)


class TileLoc(LocatorCollection):

    yaml_tag = '!Tiles'
    loc_type = 'tiles'
    cols = [
        'min_lon', 'min_lat', 'max_lon', 'max_lat',
        'cardinal_lon', 'cardinal_lat', 'abs_min_lon', 'abs_min_lat']

    def __init__(self, filename, bbox=None,
                 file_id='', dirname='', default_ext='',
                 variable='', omit_ext=False, **env):
        if not ('lat' in filename and 'lon' in filename):
            filename += '.{min_lon}.{min_lat}'

        locs = []
        meta = []
        i = 0
        for lon in range(int(bbox.min.lon), int(bbox.max.lon)):
            for lat in range(int(bbox.min.lat), int(bbox.max.lat)):
                meta_dict = {
                    'i': i,
                    'min_lon': bbox.min.lon,
                    'min_lat': bbox.min.lat,
                    'max_lon': bbox.min.lon + 1,
                    'max_lat': bbox.min.lat + 1,
                    'cardinal_lon': 'E' if bbox.min.lon >= 0 else 'W',
                    'cardinal_lat': 'N' if bbox.min.lat >= 0 else 'S',
                    'abs_min_lon': abs(bbox.min.lon),
                    'abs_min_lat': abs(bbox.min.lat)
                }
                meta.append(meta_dict.copy())
                meta_dict.update(env)
                loc = Locator(
                    file_id=file_id,
                    filename=filename, dirname=dirname, default_ext=default_ext,
                    variable=variable, omit_ext=omit_ext,
                    **meta_dict
                )
                locs.append(loc)
                i += 1

        super(TileLoc, self).__init__(locs, meta, **env)

class LayeredLoc(LocatorCollection):

    yaml_tag = '!Layers'
    loc_type = 'layers'
    cols = ['layer_min', 'layer_max']

    def __init__(self, filename, layers=[],
                 file_id='', dirname='', default_ext='',
                 variable='', omit_ext=False, **env):
        if not ('layer' in filename or 'layer' in dirname):
            filename += '.{layer_min}'

        locs = []
        meta = []
        i = 0
        for min, max in zip(layers[:-1], layers[1:]):
            meta_dict = {
                'layer_min': min,
                'layer_max': max
            }
            meta.append(meta_dict.copy())
            meta_dict.update(env)
            loc = Locator(
                file_id=file_id,
                filename=filename, dirname=dirname, default_ext=default_ext,
                variable=variable, omit_ext=omit_ext,
                **meta_dict
            )
            locs.append(loc)
            i += 1

        super(LayeredLoc, self).__init__(locs, meta, **env)


class MultipleTypeLoc(LocatorCollection):

    yaml_tag = '!MultiType'
    loc_type = 'types'
    cols = ['type']

    def __init__(self, filename, types=[],
                 file_id='', dirname='', default_ext='',
                 variable='', omit_ext=False, **env):
        if not ('type' in filename or 'type' in dirname):
            filename += '.{type}'

        locs = []
        meta = []
        i = 0
        for type in types:
            meta_dict = {
                'type': type
            }
            meta.append(meta_dict.copy())
            meta_dict.update(env)
            loc = Locator(
                file_id=file_id,
                filename=filename, dirname=dirname, default_ext=default_ext,
                variable=variable, omit_ext=omit_ext,
                **meta_dict
            )
            locs.append(loc)
            i += 1

        super(MultipleTypeLoc, self).__init__(locs, meta, **env)


class ComboLocatorCollection(LocatorCollection):

    yaml_tag = '!Combo'
    loc_type = 'combo'

    def __init__(self, filename, dimensions,
                 file_id='', dirname='', default_ext='',
                 variable='', omit_ext=False, **kwargs):
        self.env = kwargs
        self.cols = []
        locs = [None]
        meta = None
        while dimensions:
            new_locs = []
            new_meta = []
            loc_cls = Locator.__children__()[dimensions.pop(0)]
            records = [{}] if (meta is None) else meta.to_dict('records')
            for loc, rec in zip(locs, records):
                kwargs.update(rec)
                collection = loc_cls(
                    file_id=file_id,
                    filename=loc.filename if loc else filename,
                    dirname=dirname,
                    default_ext=default_ext,
                    variable=variable,
                    omit_ext=omit_ext,
                    **kwargs
                )
                self.cols.extend(collection.cols)
                new_locs.extend(collection.locs)
                # Convert dataframe to list of tuples
                for key, value in rec.items():
                    collection.meta[key] = value
                new_meta.append(collection.meta)

            locs = new_locs
            meta = pd.concat(new_meta, ignore_index=True)
        super(ComboLocatorCollection, self).__init__(
            locs, meta.to_dict('records'), **kwargs)
