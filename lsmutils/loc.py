import calendar
import datetime
import logging
import os
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

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __init__(self, file_id='', filename='', dirname='', default_ext='',
                 variable='', omit_ext=False, **env):

        self.file_id = file_id if file_id else self.file_id
        self.env = env
        self.filename = filename if filename else self.filename
        self.dirname = dirname if dirname else self.dirname
        self.default_ext = (
            default_ext if default_ext else self.default_ext)
        self.variable = (
            variable if variable else self.variable)
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
            dataset = DataFrameDataset(path, filetype=filetype)
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

    def __init__(self, file_id='', locs=[],
                 dirname='', default_ext='', variable='', omit_ext=False):
        self.file_id = file_id
        self.locs = locs

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
        return self

    def isfile(self):
        return any([loc.isfile for loc in self.locs])

    def dataset(self):
        return [loc.dataset for loc in self.locs]

    def path(self):
        return [loc.path for loc in self.locs]

    @property
    def default_ext(self):
        return self._default_ext

    @default_ext.setter
    def default_ext(self, new_ext):
        for loc in self.locs:
            loc.default_ext = new_ext
        self._default_ext = new_ext




class MonthlyLoc(LocatorCollection):

    yaml_tag = '!MonthlyFile'
    loc_type = 'monthly'

    def __init__(self, filename,
                 file_id='', dirname='', default_ext='',
                 variable='', omit_ext=False, **env):
        locs = []
        if not 'month' in filename:
            filename += '.{month:02d}'
        for month in range(1, 13):
            loc = Locator(
                file_id=file_id,
                filename=filename, dirname=dirname, default_ext=default_ext,
                variable=variable, omit_ext=omit_ext,
                month=month, month_name=calendar.month_name[month],
                month_abbr=calendar.month_abbr[month])
            locs.append(loc)

        super(MonthlyLoc, self).__init__(file_id, locs)

    def items(self):
        return {}

class Tile:

    def __init__(self, min_lon, min_lat):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = min_lon + 1
        self.max_lat = min_lat + 1
        self.cardinal_lat = 'N' if min_lat >= 0 else 'S'
        self.cardinal_lon = 'E' if min_lon >= 0 else 'W'
        self.abs_min_lon = abs(min_lon)
        self.abs_min_lat = abs(min_lat)

class TilePath(LocatorCollection):

    yaml_tag = '!Tiles'
    loc_type = 'tiles'

    def __init__(self, file_id='',
                 bbox=[], filename_fmt='', dirname='',
                 default_ext='', default_name_fmt='',
                 variable=''):
        tiles = []
        for lon in range(int(bbox[0]), int(bbox[2])):
            for lat in range(int(bbox[1]), int(bbox[3])):
                tiles += [Tile(lon, lat)]
        filename = [filename_fmt.format(**vars(tile)) for tile in tiles]

        super(TilePath, self).__init__(
                file_id=file_id,
                filename=filename, dirname=dirname,
                default_ext=default_ext,
                default_name_fmt=default_name_fmt,
                variable=variable)

        # Filter out files that don't exist
        self.filename = [self.filename[i] for i in range(len(self.path))
                             if os.path.exists(self.path[i])]
