import geopandas as gpd
import logging
from osgeo import gdal, gdal_array, ogr, osr
import netCDF4 as nc4
import numpy as np
import pandas as pd
import shapely

from math import ceil, floor

from .utils import CoordProperty, BBox

class Dataset(object):

    def __init__(self, loc):
        self.loc = loc
        self.meta = {}


class DataFrameDataset(Dataset):

    filetypes = {
        'csv': 'csv',
        'tsv': 'csv'
    }

    def __init__(self, loc, filetype='csv', **kwargs):
        self.loc = loc
        self.filetype = filetype
        self.csvargs = kwargs
        self.meta = {}
        self._dataset = None

    def new(self, data=None, index=None, columns=None):
        self._dataset = pd.DataFrame(data, index, columns)
        return self

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.load_dataframe()
        return self._dataset

    def load_dataframe(self):
        if self.filetypes[self.filetype] == 'csv':
            return pd.read_csv(self.loc.path, **self.csvargs)
        else:
            raise ValueError('Unsupported file type: {}'.format(self.filetype))

    def save(self):
        self.saveas(self.filetype)

    def saveas(self, filetype, datatype=None):
        if self.filetypes[filetype] == 'csv':
            self.dataset.to_csv(self.loc.path, **self.csvargs)


class GeoDataFrameDataset(DataFrameDataset):
    """
    Tabular data with coordinates or shapes
    """

    filetypes = {
        'csv': 'csv',
        'tsv': 'tsv'
    }

    def __init__(self, loc, filetype='csv',
                 geometry_col=None, xy_cols=None, bbox=None):
        super().__init__()
        self.geometry_col = geometry_col
        self.xy_cols = xy_cols
        self.bbox = bbox

    @property
    def dataset(self):
        """ Load data as a geopandas GeoDataFrame """

        if self._dataset is None:
            self._dataset = self.load_dataframe(self)

        if not hasattr(self, 'geometry'):
            if self.geometry_col:
                self._dataset[self.geometry_col] = (
                    self._dataset[self.geometry_col].apply(shapely.wkt.loads))
                self._dataset = gpd.GeoDataFrame(
                    self._dataset, geometry=self.geometry_col)
            elif self.xy_cols:
                self._dataset = gpd.GeoDataFrame(
                    self._dataset,
                    geometry=gpd.points_from_xy(
                        self._dataset[self.xy_cols[0]],
                        self._dataset[self.xy_cols[1]])
                )

        if self.bbox:
            self._dataset = self._dataset.cx[
                bbox.min.x:bbox.max.y, bbox.min.x:bbox.max.y]

        return self._dataset

    def crop(self, bbox):
        self.bbox = bbox


class SpatialDataset():

    def __init__(self, ds_min=None, ds_max=None, padding=None):

        if not padding is None:
            ds_min = ds_min - padding
            ds_max = ds_max + padding

        self.meta = {}

        self._min = ds_min
        self._max = ds_max
        self._center = None
        self._resolution = None
        self._ul = None
        self._ur = None
        self._ll = None
        self._lr = None
        self._bbox = None
        self._warp_output_bounds = None

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def ul(self):
        if not self._ul:
            self._ul = CoordProperty(x=self.min.x, y=self.max.y)
        return self._ul

    @property
    def ur(self):
        if not self._ur:
            self._ur = CoordProperty(x=self.max.x, y=self.max.y)
        return self._ur

    @property
    def ll(self):
        if not self._ll:
            self._ll = CoordProperty(x=self.min.x, y=self.min.y)
        return self._ll

    @property
    def lr(self):
        if not self._lr:
            self._lr = CoordProperty(x=self.max.x, y=self.min.y)
        return self._lr

    @property
    def bbox(self):
        if not self._bbox:
            self._bbox = BBox(llc=self.ll, urc=self.ur)
        return self._bbox

    @property
    def warp_output_bounds(self):
        if not self._warp_output_bounds:
            self._warp_output_bounds = [self.min.x, self.min.y,
                                        self.max.x, self.max.y]
        return self._warp_output_bounds

    @property
    def rev_warp_output_bounds(self):
        if not self._rev_warp_output_bounds:
            self._rev_warp_output_bounds = [self.cmin.x, self.cmin.y,
                                            self.cmax.x, self.cmax.y]
        return self._rev_warp_output_bounds

    @property
    def center(self):
        if not self._center:
            self._center = (self.max + self.min) / 2.
        return self._center

class NetCDFDataset(SpatialDataset):

    filetypes = {
        'nc': 'nc'
    }

    xdims = ['x', 'X', 'lon', 'longitude']
    ydims = ['y', 'Y', 'lat', 'latitude']

    def __init__(self, loc):
        self.loc = loc
        self.meta = {}
        self.mod = 'r'

        super().__init__()

        self._cgrid = None
        self._cmin = None
        self._cmax = None
        self._coords = None
        self._dataset = None
        self._resolution = None
        self._size = None
        self._xdim = None
        self._ydim = None

    @property
    def dataset(self):
        if not self._dataset:
            self._dataset = nc4.Dataset(self.loc.path, mode=self.mod)
        return self._dataset

    @property
    def xdim(self):
        if not self._xdim:
            self._xdim = [
                dim for dim in self.dataset.dimensions
                if dim in self.xdims][0]
        return self._xdim

    @property
    def ydim(self):
        if not self._ydim:
            self._ydim = [
                dim for dim in self.dataset.dimensions
                if dim in self.ydims][0]
        return self._ydim

    @property
    def cgrid(self):
        if not self._cgrid:
            self._cgrid = CoordProperty(
                self.dataset[self.xdim][:],
                self.dataset[self.ydim][:])
        return self._cgrid

    @property
    def coords(self):
        if not self._coords:
            xx, yy = np.meshgrid(self.cgrid.x, self.cgrid.y)
            self._coords = [
                CoordProperty(x, y)
                for x, y in zip(xx.flatten(), yy.flatten())
            ]
        return self._coords

    @property
    def cmin(self):
        if not self._cmin:
            self._cmin = CoordProperty(min(self.cgrid.x), min(self.cgrid.y))
        return self._cmin

    @property
    def cmax(self):
        if not self._cmax:
            self._cmax = CoordProperty(max(self.cgrid.x), max(self.cgrid.y))
        return self._cmax

    @property
    def max(self):
        if not self._max:
            self._max = CoordProperty(
                self.cmax.x + self.res.x / 2,
                self.cmax.y + self.res.y / 2 )
        return self._max

    @property
    def min(self):
        if not self._min:
            self._min = CoordProperty(
                self.cmin.x - self.res.x / 2,
                self.cmin.y - self.res.y / 2 )
        return self._min

    @property
    def center(self):
        if not self._center:
            self._center = (self.cmax + self.cmin) / 2.
        return self._center

    @property
    def resolution(self):
        if not self._resolution:
            self._resolution = CoordProperty(
                x=abs(np.mean(self.cgrid.x[1:] - self.cgrid.x[:-1])),
                y=abs(np.mean(self.cgrid.y[1:] - self.cgrid.y[:-1]))
            )
        return self._resolution

    @property
    def res(self):
        return self.resolution

    @property
    def size(self):
        if not self._size:
            self._size = CoordProperty(
                x=self.dataset.dimensions[self.xdim].size,
                y=self.dataset.dimensions[self.ydim].size)
        return self._size


class GDALDataset(SpatialDataset):

    filetypes = {
        'nc': 'netCDF',
        'asc': 'AAIGrid',
        'gtif': 'GTiff',
        'tif': 'GTiff',
        'hgt': 'SRTMHGT',
        'hdf': 'HDF5',
        'vrt': 'VRT'
    }

    prefixes = {
        'nc': 'NETCDF',
        'hdf': 'HDF4_EOS:EOS_GRID'
    }

    def __init__(self, loc, template=None, filetype='gtif'):
        self.loc = loc
        self.filetype = filetype
        self.meta = {}

        # Initialize internal attributes
        self._dataset = None
        self._geotransform = None
        self._nodata = None
        self._datatype = None
        self._srs = None
        self._resolution = None
        self._size = None
        self._array = None
        self._masked = None
        self._min = None
        self._max = None
        self._cmin = None
        self._cmax = None
        self._cgrid = None
        self._gridcorners = None
        self._grid = None
        self._ul = None
        self._ur = None
        self._ll = None
        self._lr = None
        self._bbox = None
        self._warp_output_bounds = None
        self._rev_warp_output_bounds = None

        # Reopen file if updating
        self.mod = gdal.GA_ReadOnly

        # New dataset with template metadata
        if template:
            driver = gdal.GetDriverByName(self.filetypes[filetype])
            self._dataset = driver.CreateCopy(loc.path, template.dataset)


    @property
    def dataset(self):
        gdal_path = 'unknown file'

        if not self._dataset:
            if not hasattr(self.loc, 'file_id'):
                gdal_path = self.loc
            elif not self.loc.variable:
                gdal_path = self.loc.path
            else:
                gdal_path = '{filetype}:{path}:{variable}'.format(
                    filetype = self.prefixes[self.filetype],
                    path = self.loc.path,
                    variable = self.loc.variable)
            self._dataset = gdal.Open(gdal_path, self.mod)

        if not self._dataset:
            logging.warning('%s did not load as raster' % gdal_path)

        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self.__init__(self.loc, filetype=self.filetype)
        self._dataset = dataset

    @dataset.deleter
    def dataset(self):
        self._dataset = None

    def save(self):
        self.saveas('gtif')

    def saveas(self, filetype, datatype=None):
        if datatype is None:
            datatype = self.datatype
        if filetype != self.filetype:
            gdal.Translate(self.loc.ext(filetype),
                           self.dataset,
                           format=self.filetypes[filetype])
            logging.info('Translated to {ext} with datatype {dt}'.format(
                    ext=filetype, dt=datatype))
        return GDALDataset(self.loc, filetype=filetype)

    def chmod(self, mod):
        if self.mod == mod:
            return self

        self.dataset.FlushCache()
        self._dataset = None
        self._dataset = gdal.Open(self.loc.path, mod)
        if not self.dataset:
            logging.error('Dataset at %s did not load for writing',
                          self.loc.path)
        self.mod = mod
        return self

    @property
    def geotransform(self):
        if not self._geotransform:
            self._geotransform = self.dataset.GetGeoTransform()
        return self._geotransform

    @property
    def srs(self):
        if not self._srs:
            wkt = self.dataset.GetProjection()
            self._srs = osr.SpatialReference()
            self._srs.ImportFromWkt(wkt)
        return self._srs

    @srs.setter
    def srs(self, new_espg):
        logging.debug('Trying projection {}'.format(self.espg))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(new_espg)
        self._dataset.SetProjection(srs.ExportToWkt())
        self._srs = srs

    @property
    def datatype(self):
        if not self._datatype:
            try:
                self._datatype=self.dataset.GetRasterBand(1).DataType
            except AttributeError:
                self._datatype=None
        return self._datatype

    @property
    def nodata(self):
        if not self._nodata:
            self._nodata = self.dataset.GetRasterBand(1).GetNoDataValue()
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata):
        self.chmod(gdal.GA_Update)
        old_nodata = self.nodata
        self.dataset.GetRasterBand(1).SetNoDataValue(new_nodata)
        array = self.array
        array[array==old_nodata] = new_nodata
        self.array = array

        self.dataset.FlushCache()
        self._masked = None
        self._dataset = None
        self._nodata = self.dataset.GetRasterBand(1).GetNoDataValue()

    @property
    def resolution(self):
        if not self._resolution:
            self._resolution = CoordProperty(x=abs(self.geotransform[1]),
                                             y=abs(self.geotransform[5]))
        return self._resolution

    @property
    def res(self):
        return self.resolution

    @property
    def array(self):
        if self._array is None:
            self._array = self.dataset.GetRasterBand(1).ReadAsArray()
            if self.geotransform[1] < 0:
                self._array = np.fliplr(self._array)
            if self.geotransform[5] < 0 :
                self._array = np.flipud(self._array)
        return self._array

    @array.setter
    def array(self, new_array):
        # Allow updates to the array
        self.chmod(gdal.GA_Update)

        # Deal with negative 'resolution' values
        if self.geotransform[1] < 0:
            new_array = np.fliplr(new_array)
        if self.geotransform[5] < 0 :
            new_array = np.flipud(new_array)

        # Set datatype to match the new array if necessary
        ds = self.dataset
        nodata = self.nodata

        # GDAL doesn't have 64-bit integers
        if new_array.dtype == np.int64:
            new_array = new_array.astype(np.int32)
        new_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(new_array.dtype)
        if not new_dtype == ds.GetRasterBand(1).DataType:
            geotransform = ds.GetGeoTransform()
            srs = ds.GetProjection()
            driver = ds.GetDriver()
            ds = driver.Create(self.loc.path,
                               new_array.shape[1],
                               new_array.shape[0],
                               1, new_dtype)
            ds.SetGeoTransform(geotransform)
            ds.SetProjection(srs)

        # Write new array
        ds.GetRasterBand(1).WriteArray(new_array)
        # Nodata value does not transfer
        if nodata:
            ds.GetRasterBand(1).SetNoDataValue(nodata)

        # Clean up
        ds.FlushCache()
        self._dataset = ds
        self._array = None

    @property
    def masked(self):
        if self._masked is None:
            self._masked = np.ma.MaskedArray(
                    self.array, mask=self.array == self.nodata)
            np.ma.set_fill_value(self._masked, self.nodata)
        return self._masked

    @masked.setter
    def masked(self, new_ma):
        self.array = self.masked.filled()

    @property
    def size(self):
        if not self._size:
            shape = self.array.shape
            self._size = CoordProperty(x=shape[1], y=shape[0])
        return self._size

    @property
    def min(self):
        if not self._min:
            self.max
        return self._min

    @property
    def max(self):
        if not self._max:
            res_x = self.geotransform[1]
            res_y = self.geotransform[5]

            x_bounds = (self.geotransform[0] + res_x * self.size.x,
                        self.geotransform[0])
            y_bounds = (self.geotransform[3] + res_y * self.size.y,
                        self.geotransform[3])
            self._max = CoordProperty(x=max(x_bounds), y=max(y_bounds))
            self._min = CoordProperty(x=min(x_bounds), y=min(y_bounds))
        return self._max

    @property
    def cmax(self):
        if not self._cmax:
            res_x = self.geotransform[1]
            res_y = self.geotransform[5]

            x_bounds = (self.geotransform[0] + res_x * self.size.x,
                        self.geotransform[0])
            y_bounds = (self.geotransform[3] + res_y * self.size.y,
                        self.geotransform[3])
            self._cmax = CoordProperty(x=max(x_bounds),
                                      y=max(y_bounds))
            self._cmin = CoordProperty(x=min(x_bounds),
                                      y=min(y_bounds))
        return self._cmax

    @property
    def cmin(self):
        if not self._cmin:
            self.cmax
        return self._cmin

    @property
    def grid(self):
        if self._grid is None:
            xgrid = np.linspace(self.min.x + .5 * self.res.x,
                                self.max.x - .5 * self.res.x,
                                self.size.x + 1)
            ygrid = np.linspace(self.min.y + .5 * self.res.y,
                                self.max.y - .5 * self.res.y,
                                self.size.y + 1)
            self._grid = CoordProperty(x=xgrid, y=ygrid)
        return self._grid

    @property
    def cgrid(self):
        if self._cgrid is None:
            xgrid = np.linspace(self.cmin.x,
                                self.cmax.x,
                                self.size.x)
            ygrid = np.linspace(self.cmin.y,
                                self.cmax.y,
                                self.size.y)
            self._cgrid = CoordProperty(x=xgrid, y=ygrid)
        return self._cgrid

    def gridmin(self, res):
        return self.min.grid_coord(res, method=floor)

    def gridmax(self, res):
        return self.max.grid_coord(res, method=ceil)

    def gridcmin(self, res):
        return self.cmin.grid_coord(res, method=floor)

    def gridcmax(self, res):
        return self.cmax.grid_coord(res, method=ceil)

    def gridcorners(self, res, padding=None):
        return SpatialDataset(
                ds_min=self.gridcmin(res),
                ds_max=self.gridcmax(res),
                padding=padding)

    def get_value(self, coord):
        xi = floor((coord.x - self.min.x) / self.res.x)
        yi = floor((coord.y - self.min.y) / self.res.y)
        return self.array[yi, xi]


class BoundaryDataset(SpatialDataset):

    def __init__(self, loc, update=False, driver='ESRI Shapefile'):
        self.loc = loc
        self.update = update
        self.driver = driver
        self.meta = {}
        self._dataset = None
        self._srs = None
        self._layers = None
        self._min = None
        self._max = None
        self._ul = None
        self._ur = None
        self._ll = None
        self._lr = None
        self._bbox = None

    def new(self):
        driver = ogr.GetDriverByName(self.driver)
        self._dataset = driver.CreateDataSource(self.loc.path)
        return self

    def close(self):
        self.dataset.Destroy()

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = ogr.Open(self.loc.path, update=self.update)
        if self._dataset is None:
            logging.debug('Dataset at %s did not load as shapefile',
                          self.loc.shp)
        return self._dataset

    @dataset.deleter
    def dataset(self):
        self._dataset = None

    @property
    def srs(self):
        if not self._srs:
            self._srs = self.layers[0].GetSpatialRef()
        return self._srs

        @srs.setter
        def srs(self, new_espg):
            logging.debug('Trying projection {}'.format(self.espg))
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(new_espg)
            self._dataset.SetProjection(srs.ExportToWkt())
            self._srs = srs

    @property
    def layers(self):
        if not self._layers:
            self._layers = []
            n = self.dataset.GetLayerCount()
            for i in range(0, n):
                self._layers.append(self.dataset.GetLayerByIndex(i))
        return self._layers

    @property
    def min(self):
        if not self._min:
            layer = self.dataset[0]
            (minx, maxx, miny, maxy) = layer.GetExtent()
            self._min = CoordProperty(x=minx, y=miny)
            self._max = CoordProperty(x=maxx, y=maxy)
        return self._min

    @property
    def max(self):
        if not self._max:
            self.min
        return self._max

    def gridmin(self, res):
        return self.min.grid_coord(res, method=floor)

    def gridmax(self, res):
        return self.max.grid_coord(res, method=ceil)

    def gridcorners(self, res, padding=None):
        return Dataset(ds_min=self.gridmin(res), ds_max=self.gridmax(res),
                       padding=padding)

    def chmod(self, update):
        self.update = update
        self._dataset = None

    def plot(self, *args, **kwargs):
        pass
