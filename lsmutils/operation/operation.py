import copy
import csv
import datetime
import geopandas as gpd
import glob
import inspect
import logging
import netCDF4 as nc4
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units
import multiprocessing
import numpy as np
import os
from osgeo import gdal, ogr, osr
import pandas as pd
import random
import re
import shutil
import string
import subprocess
import yaml
import collections

from lsmutils.loc import *
from lsmutils.dataset import GDALDataset, BoundaryDataset, DataFrameDataset
from lsmutils.utils import CoordProperty, BBox

OutputType = collections.namedtuple(
    'OutputType', ['key', 'filetype', 'loctype'])
OutputType.__new__.__defaults__ = ('', '', 'file')

class OperationMeta(yaml.YAMLObjectMetaclass):

    def __children__(cls):
        children = {}
        for child in cls.__subclasses__():
            children.update({child.name: child})
            children.update(child.__children__())
        return children


class Operation(yaml.YAMLObject, metaclass=OperationMeta):
    """
    A parent class to perform some operation on a raster dataset

    Child classes must define at minimum:
      - title: a human-readable name,
      - name: a path-friendly identifier
      - output_types: a list to parse in to locators
        for output files
      - run(): a function to perform the raster operation.
    """

    yaml_tag = u'!Operation'

    title = NotImplemented
    name = NotImplemented
    output_types = NotImplemented
    filename_format = '{output_label}'

    end_msg = '{title} saved at: \n    {loc}'
    error_msg = '{title} calculation FAILED'

    def __init__(self, inpt, out, dims=[]):
        self.inpt = inpt
        self.out = out
        self.dims = dims

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        fields['inpt'] = fields.pop('in')
        child_cls = cls.__children__()[fields.pop('name')]
        return child_cls(**fields)

    def configure(self, cfg, locs={}, scripts={}, **kwargs):
        logging.debug('Configuring %s operation', self.name)

        # Extract configuration variables
        self.kwargs = {
            key.replace('-', '_'): value for key, value in kwargs.items()}
        self.cfg = cfg
        self.log_level = cfg['log_level']
        self.case_id = cfg['case_id']
        self.base_dir = cfg['base_dir']
        if 'nprocesses' in cfg:
            self.nprocesses = cfg['nprocesses']
        else:
            self.nprocesses = os.cpu_count()

        self.scripts = scripts

        # Get non-automatic attributes for filename
        self.attributes = kwargs.copy()
        self.attributes.update(
            {title: getattr(self, title) for title in dir(self)})

        logging.debug('Formatting %s file names', self.name)
        self.filenames = {
            ot.key: self.filename_format.format(
                output_label=self.get_label(ot.key),
                **self.attributes)
            for ot in self.output_types}

        logging.debug('Resolving %s locs', self.name)
        if not self.dims:
            self.locs = {
                ot.key: Locator.__children__()[ot.loctype](
                    filename=self.filenames[ot.key],
                    default_ext=ot.filetype).configure(cfg)
                for ot in self.output_types}
        else:
            env = {}
            meta = None
            for loc in self.kwargs.values():
                if hasattr(loc, 'env'):
                    env.update(loc.env)
                if hasattr(loc, 'loc_type'):
                    if loc.loc_type in ['regex', 'list', 'tiles']:
                        env.update({'meta': loc.meta.to_dict('records')})

            if 'meta' in env:
                self.locs = {
                    ot.key: ListLoc(
                        filename=self.filenames[ot.key],
                        default_ext=ot.filetype,
                        **copy.deepcopy(env)
                        ).configure(cfg, file_id=ot.key)
                    for ot in self.output_types
                }
                # Group by dimensions and reduce
                for loc in self.locs.values():
                    loc.reduce(self.dims)
            else:
                env.update({'dimensions': self.dims.copy()})
                self.locs = {
                    ot.key: ComboLocatorCollection(
                        filename=self.filenames[ot.key],
                        default_ext=ot.filetype,
                        **copy.deepcopy(env)
                        ).configure(cfg)
                    for ot in self.output_types
                }

        # Save files in configured location, not default
        for key, label in self.out.items():
            if label in locs:
                self.locs[key].filename = locs[label].filename
                self.locs[key].dirname = locs[label].dirname

        for key, loc in self.locs.items():
            logging.debug('%s path at %s', key, loc)

        # Initialize variables
        self.datasets = []
        return self

    def get_label(self, output_type):
        if output_type in self.out:
            return self.out[output_type]
        else:
            idstr = ''.join([
                random.choice(string.ascii_letters + string.digits)
                for n in range(6)])
            return output_type + '_' + idstr

    def relabel(self, new_labels):
        for pykey, dsname in self.inpt.items():
            if str(dsname) in new_labels:
                self.inpt[pykey] = new_labels[dsname]
                logging.debug(
                    'Relabelled input %s to %s', dsname, new_labels[dsname])

        for pykey, dsname in self.out.items():
            if str(dsname) in new_labels:
                self.out[pykey] = new_labels[dsname]
                logging.debug(
                    'Relabelled output %s to %s', dsname, new_labels[dsname])

    def run(self, **kwargs):
        raise NotImplementedError

    def save(self):
        logging.info('Calculating %s with data:', self.title)
        for key, value in self.kwargs.items():
            if hasattr(value, 'path'):
                logging.info('    %s\n    %s', key, value)
            else:
                logging.info('    %s\n    %s', key, value)

        # Perform raster operation
        dim_values = None
        if self.dims:
            self.full_locs = copy.copy(self.locs)
            self.full_locs.update(self.kwargs)

            # Get parameter combinations from locator collections
            for key, loc in self.full_locs.items():
                if hasattr(loc, 'cols'):
                    loc_dim_values = loc.get_dim_values(self.dims)
                    if not dim_values is None:
                        dim_values = dim_values.join(
                            loc_dim_values, lsuffix='.copy')
                    else:
                        dim_values = loc_dim_values

        if not dim_values is None:
            print(dim_values)
            # Run operation for each combination in parallel
            pool = multiprocessing.Pool(self.nprocesses)
            pool.map(self.run_subset, dim_values.to_dict('records'))

        else:
            # Run single operation with no dimensions
            kwargs = copy.copy(self.kwargs)
            kwargs.update({
                key: loc.dataset for key, loc in kwargs.items()
                if hasattr(loc, 'dataset')})
            self.run(**kwargs)

        # Report status
        for key, loc in self.locs.items():
            logging.info(self.end_msg.format(title=self.title, loc=loc))

        return self.locs

    def run_subset(self, dim_row):
        subset_kwargs = copy.copy(self.kwargs)

        # Subset locators
        for key, loc in self.full_locs.items():
            if hasattr(loc, 'file_id'):
                ds_name = loc.file_id
                if ds_name in dim_row:
                    if key in self.kwargs:
                        subset_kwargs[key] = loc.get_subset(dim_row[ds_name])
                    else:
                        self.locs[key] = loc.get_subset(dim_row[ds_name])

        # Replace dataset locators with datasets
        subset_kwargs.update({
            key: loc.dataset for key, loc in subset_kwargs.items()
            if hasattr(loc, 'dataset')})

        self.run(**subset_kwargs)

        # Restore locators
        self.locs = copy.copy(self.full_locs)

        # Remove datasets from memory
        for loc in subset_kwargs.values():
            del loc


class SumOp(Operation):

    title = 'Sum raster'
    name = 'sum'
    output_type = 'csv'

    def run(self, input_ds, label_ds=None, fraction_ds=None):
        array = np.ma.masked_where(input_ds.array==input_ds.nodata,
                                   input_ds.array)

        if not fraction_ds is None:
            fractions = np.ma.masked_where(input_ds.array==input_ds.nodata,
                                           fraction_ds.array)
            array = array * fractions

        sum = array.sum()

        if label_ds is None:
            label = input_ds.loc.filename
        else:
            label = label_ds.loc.filename

        with open(self.path.csv, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([label, sum])


class MaskOp(Operation):

    title = 'Raster mask of boundary area'
    name = 'mask'

    def run(self, input_ds, boundary_ds):
        output_ds = GDALDataset(self.path, template=input_ds)
        output_ds.dataset.GetRasterBand(1).Fill(-9999)
        # Rasterize the shapefile layer to our new dataset as 1's
        status = gdal.RasterizeLayer(
                output_ds.dataset,  # output to new dataset
                [1],  # first band
                boundary_ds.dataset[0],  # rasterize first layer
                None, None,  # no transformations we're in same projection
                [1],  # burn value 1
                ['ALL_TOUCHED=TRUE'],
        )
        output_ds.nodata = -9999

class Slope(Operation):
    """
    Compute Slope and Aspect from Elevation
    """

    title = 'Slope and Aspect'
    name = 'slope'
    output_types = [
        OutputType('slope', 'tif'),
        OutputType('aspect', 'tif')
    ]

    def run(self, elevation_ds):
        gdal.DEMProcessing(
            self.locs['slope'].path, elevation_ds.dataset, 'slope')
        gdal.DEMProcessing(
            self.locs['aspect'].path, elevation_ds.dataset, 'aspect')


class RasterToShapefileOp(Operation):
    """ Convert a raster to a shapefile """

    title = 'Raster to Shapefile'
    name = 'raster-to-shapefile'
    output_types = [OutputType('shapefile', 'shp')]

    def run(self, input_ds):
        # Convert the input raster to 0 and 1 mask
        # (A hack to make up for poor labeling by LabelGages)
        array = input_ds.array
        array[array==0] = 1
        array[array!=1] = 0
        input_ds.array = array

        # Add raster boundary to shapefile
        input_band = input_ds.dataset.GetRasterBand(1)
        output_ds = BoundaryDataset(self.locs['shapefile'], update=True).new()
        output_layer = output_ds.dataset.CreateLayer(
            'polygonized_raster', srs = input_ds.srs)
        gdal.Polygonize(input_band, input_band,
                        output_layer, -1, [], callback=None)

        # Clean up
        output_ds.close()


class LatLonToShapefileOp(Operation):
    """ Create a shapefile with a single point """

    title = 'Coordinate to Shapefile'
    name = 'coordinate-to-shapefile'
    output_types = [OutputType('shapefile', 'shp')]

    def run(self, coordinate, idstr='coordinate'):
        # Get spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(coordinate.epsg)

        # Create layer
        shp_ds = BoundaryDataset(self.locs['shapefile'], update=True).new()
        output_layer = shp_ds.dataset.CreateLayer(
                idstr, srs=srs, geom_type=ogr.wkbPoint)

        # Create geometry
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(float(coordinate.lon), float(coordinate.lat))
        point.AssignSpatialReference(srs)

        # Add to feature
        output_feature = ogr.Feature(output_layer.GetLayerDefn())
        output_feature.SetGeometry(point)
        output_layer.CreateFeature(output_feature)

        # Clean up'
        del shp_ds.dataset

class MeltNetCDF(Operation):
    '''
    Generates a CSV file with values for each coordinate in a raster
    '''
    title = 'Melted Raster'
    name = 'melt-nc'
    output_type = 'melted_raster'

    def run(self, path, raster_ds, variable):
        with open(raster_ds.loc.nc) as raster_file:
            netcdf = scipy.io.netcdf.netcdf_file(raster_file, 'r')
            lat = netcdf.lat
            lon = netcdf.lon
            data = getattr(netcdf, variable)
        lats, lons = np.meshgrid(lat, lon)
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        data_flat = data.flatten()

        with open(self.path.csv, 'a') as csv_file:
            writer = csv.writer(csv_file)
            for i in range(lats_flat):
                writer.write([lats_flat[i], lons_flat[i], data_flat[i]])


class Melt(Operation):
    '''
    Generates a CSV file with values for each coordinate in a raster
    '''
    title = 'Melted Raster'
    name = 'melt'
    output_type = 'melted_raster'

    def run(self, path, raster_ds):
        data = raster_ds.array
        lons, lats = np.meshgrid(raster_ds.cgrid.lon, raster_ds.cgrid.lat)

        lons_flat = lons.flatten()
        lats_flat = lats.flatten()
        data_flat = data.flatten()

        with open(self.path.csv, 'a') as csv_file:
            writer = csv.writer(csv_file)
            for row in zip(lats_flat, lons_flat, data_flat):
                if not np.isnan(row[2]):
                    writer.writerow(row)

class WriteOneVarNetCDFOp(Operation):
    """ Write datasets to a netcdf file """

    title = 'Write NetCDF'
    name = NotImplemented
    output_types = [OutputType('netcdf', 'nc')]
    varname = NotImplemented
    dtype = NotImplemented
    units = NotImplemented
    nc_format='NETCDF4_CLASSIC'
    fill = NotImplemented

    def run(self, input_ds, uniform_value=False):
        ncfile = nc4.Dataset(
                self.locs['netcdf'].path, 'w', format=self.nc_format)
        ncfile.history = 'Created using lsmutils {}'.format(
                datetime.datetime.now())

        array = input_ds.array
        nodata = input_ds.nodata
        array[array==nodata] = self.fill

        if uniform_value:
            array[:] = uniform_value

        # Add dimensions
        t_dim = ncfile.createDimension("time", None)
        t_var = ncfile.createVariable("time", "f8", ("time",))
        t_var.units = 'time'
        t_var[:] = np.array([0])

        lons = input_ds.cgrid.lon
        lon_dim = ncfile.createDimension("x", len(lons))
        lon_var = ncfile.createVariable("x", "f8", ("x",))
        lon_var.standard_name = 'projection_x_coordinate'
        lon_var.units = 'Meters'
        lon_var[:] = lons

        lats = input_ds.cgrid.lat
        lat_dim = ncfile.createDimension("y", len(lats))
        lat_var = ncfile.createVariable("y", "f8", ("y",))
        lat_var.standard_name = 'projection_y_coordinate'
        lat_var.units = 'Meters'
        lat_var[:] = lats

        # Add variable of interest
        ncvar = ncfile.createVariable(
                self.varname, self.dtype, ('time', 'y', 'x'))
        ncvar.missing_value = self.fill
        ncvar.units = self.units
        ncvar[0:1,:,:] = np.expand_dims(array, axis=0)

        ncfile.close()

        self.locs['netcdf'].netcdf_variable = self.varname

class WriteUniformNetCDFOp(Operation):
    """ Write datasets to a netcdf file """

    title = 'Write Uniform NetCDF'
    name = NotImplemented
    output_types = [OutputType('netcdf', 'nc')]
    varnames = NotImplemented
    layers = NotImplemented
    dtypes = NotImplemented
    units = NotImplemented
    nc_format='NETCDF4_CLASSIC'
    fills = NotImplemented

    def run(self, template_ds):
        ncfile = nc4.Dataset(
                self.locs['netcdf'].path, 'w', format=self.nc_format)
        ncfile.history = 'Created using lsmutils {}'.format(
                datetime.datetime.now())

        # Add dimensions
        t_dim = ncfile.createDimension("time", None)
        t_var = ncfile.createVariable("time", "f8", ("time",))
        t_var.units = 'time'
        t_var[:] = np.array([0])

        lons = template_ds.cgrid.lon
        lon_dim = ncfile.createDimension("x", len(lons))
        lon_var = ncfile.createVariable("x", "f8", ("x",))
        lon_var.standard_name = 'projection_x_coordinate'
        lon_var.units = 'Meters'
        lon_var[:] = lons

        lats = template_ds.cgrid.lat
        lat_dim = ncfile.createDimension("y", len(lats))
        lat_var = ncfile.createVariable("y", "f8", ("y",))
        lat_var.standard_name = 'projection_y_coordinate'
        lat_var.units = 'Meters'
        lat_var[:] = lats

        # Add variables
        for name, layers, dtype, units, fill in zip(
            self.varnames, self.layers, self.dtypes, self.units, self.fills):
            for layer in range(layers):
                varname = name.format(layer=layer) if 'layer' in name else name
                ncvar = ncfile.createVariable(
                        varname, dtype, ('time', 'y', 'x'))
                ncvar.units = units
                ncvar[:,:] = np.expand_dims(
                    np.ones((template_ds.size.x, template_ds.size.y))*fill,
                    axis=0)

        ncfile.close()
