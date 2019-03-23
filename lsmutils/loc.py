import datetime
import logging
import os
import yaml

from .dataset import GDALDataset, BoundaryDataset

class Path(yaml.YAMLObject):
    """
    A class to keep track of file structures
    """
    yaml_tag = '!File'

    file_id = ''
    filename = ''
    dirname = '{temp_dir}'
    default_ext = ''
    default_name_fmt='{basin_id}_{file_id}'
    netcdf_variable = ''

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __init__(self, file_id='',
                 filename='', dirname='',
                 default_ext='', default_name_fmt='',
                 netcdf_variable=''):
        """
        Set path variables
        """
        
        self.file_id = file_id if file_id else self.file_id
        self.env = None
        self.filename = filename if filename else self.filename
        self.dirname = dirname if dirname else self.dirname
        self.default_ext = (
            default_ext if default_ext else self.default_ext)
        self.default_name_fmt = (
            default_name_fmt if default_name_fmt else self.default_name_fmt)
        self.netcdf_variable = (
            netcdf_variable if netcdf_variable else self.netcdf_variable)

    def configure(self, cfg, file_id='', dirname=''):
        self.file_id = file_id if file_id else self.file_id
        self.env = cfg
        self.env['file_id'] = self.file_id

        # Set base directory
        if 'base_dir' in cfg:
            self.base_dir = cfg['base_dir']
        else:
            self.base_dir = os.getcwd()

        # Render filename variables
        try:
            self.filename = self.filename.format(**cfg)
        except AttributeError:
            self.filename = [fn.format(**cfg) for fn in self.filename]

        # Render directory name
        self.dirname = dirname if dirname else self.dirname
        self.dirname = self.dirname.format(**cfg)
        if not os.path.isabs(self.dirname):
            self.dirname = os.path.join(self.base_dir, self.dirname)

        # Create directory if it does not exist
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        return self
    
    @property
    def default_name(self):
        time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        return self.default_name_fmt.format(time=time, **self.env)
        
    def _path(self, filename, extension):
        if extension:
            try:
                ext_name = '.'.join([filename, extension])
                path = os.path.join(self.dirname, ext_name)
            except TypeError:
                ext_name = ['.'.join([fn, extension]) for fn in filename]
                path = [os.path.join(self.dirname, bn) for bn in ext_name]
        else:
            try:
                path = os.path.join(self.dirname, filename)
            except TypeError:
                path = [os.path.join(self.dirname, fn) for fn in filename]
        return path
    
    def ext(self, extension):        
        try:
            full_path = self._path(self.filename, extension)
        except AttributeError:
            full_path = [self._path(fn, extension) for fn in self.filename]
        return full_path

    @property
    def path(self):
        if self.default_ext:
            return self.ext(self.default_ext)
        return self.no_ext

    @property
    def exists(self):
        try:
            exists = os.path.exists(self.path)
        except TypeError:
            # Allow a list of paths
            exists = all([os.path.exists(path) for path in self.path])
        return exists
    
    @property
    def isfile(self):
        try:
            isfile = os.path.isfile(self.path)
        except TypeError:
            # Allow a list of paths
            is_file_list = [os.path.isfile(path) for path in self.path]
            if not all(is_file_list):
                unfiles = [pth for pth in self.path if not os.path.isfile(pth)]
                for path in unfiles:
                    logging.debug('{} is not a file'.format(path))
                return False
            return True
        return isfile
    
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
        try:
            # Don't bother checking for multiple files
            # if this is already a file
            os.path.isfile(self.path)
            self._dataset = self.get_dataset(self)
            return self._dataset
        except Exception:
            logging.debug('{} is not a file; trying list'.format(self.path))
        
        # Try list
        try:
            path_list = [
                Path(filename=fn, dirname=self.dirname,
                     default_ext=self.default_ext).configure(self.env)
                for fn in self.filename]
            self._dataset = [self.get_dataset(pth) for pth in path_list]
            if not any([ds.dataset is None for ds in self._dataset]):
                return self._dataset
        except AttributeError:
            logging.debug('Unable to load dataset {} as a list'.format(
                          self.file_id))
            return None
        
    def get_dataset(self, path):
        try:
            os.path.isfile(path.path)
        except AttributeError:
            return None

        # Try OGR if the file has a shapefile extension
        if self.default_ext=='shp':
            try:
                dataset = BoundaryDataset(path)
                if not dataset.dataset is None:
                    return dataset
            except Exception:
                dataset = None
        
        # Try raster
        try:
            filetype = self.default_ext if self.default_ext else ''
            dataset = GDALDataset(path, filetype=filetype)
            if not dataset.dataset is None:
                return dataset
            print('{} failed to load as raster'.format(path.path))
        except Exception as exc:
            dataset = None
            print(exc)
            print('{} failed to load as raster'.format(path.path))
        return None

    def __getitem__(self, key):
        new_path = Path(
                file_id=self.file_id,
                filename=self.filename[key],
                dirname=self.dirname,
                default_ext=self.default_ext,
                default_name_fmt=self.default_name_fmt,
                netcdf_variable=self.netcdf_variable)
        if self.env:
            new_path.configure(self.env)
        return new_path

    def __iter__(self):
        raise NotImplementedError

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

class TilePath(Path):

    yaml_tag = '!Tiles'

    def __init__(self, file_id='',
                 bbox=[], filename_fmt='', dirname='',
                 default_ext='', default_name_fmt='',
                 netcdf_variable=''):
        tiles = []
        for lon in range(int(bbox[0]), int(bbox[2])):
            for lat in range(int(bbox[1]), int(bbox[3])):
                tiles += [Tile(lon, lat)]
        filename = [filename_fmt.format(**vars(tile)) for tile in tiles]
        print(filename)
        super(TilePath, self).__init__(
                file_id=file_id,
                filename=filename, dirname=dirname,
                default_ext=default_ext,
                default_name_fmt=default_name_fmt,
                netcdf_variable=netcdf_variable)

        # Filter out files that don't exist
        self.filename = [self.filename[i] for i in range(len(self.path))
                             if os.path.exists(self.path[i])]


class DirPath(Path):

    yaml_tag = '!Directory'
    default_name_fmt = ''
    default_ext = None

# Location of C scripts for generating RVIC files
class ScriptPath(DirPath):

    dirname = 'src_setup'
    default_ext = None

class TemplatePath(DirPath):

    file_id = 'templates'
    dirname = 'templates'
