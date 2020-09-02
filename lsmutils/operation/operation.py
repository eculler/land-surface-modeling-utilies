import copy
import csv
import datetime
import gdal
import geopandas as gpd
import glob
import inspect
import logging
import netCDF4 as nc4
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units
import multiprocessing
import numpy as np
import ogr
import os
import osr
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


class UniqueOp(Operation):

    title = 'A unique number in each VIC-resolution grid cell'
    name = 'unique'
    output_type = 'unique'
    filename_format = '{case_id}_{output_type}'

    def run(self, input_ds):
        # Generate unique dataset
        unique_ds = GDALDataset(self.path, template=input_ds)
        unique_ds.array = np.arange(
                1,
                unique_ds.array.size + 1
        ).reshape(unique_ds.array.shape)


class ZonesOp(Operation):

    title = 'Downscaled Zones for each VIC gridcell'
    name = 'zones'
    output_type = 'zones'
    filename_format = '{case_id}_{output_type}'

    def run(self, unique_ds, fine_ds):
        res_warp_options = gdal.WarpOptions(
                xRes = fine_ds.res.x,
                yRes = fine_ds.res.y,
                outputBounds = fine_ds.rev_warp_output_bounds,
                resampleAlg='average')
        gdal.Warp(path, unique_ds.dataset, options=res_warp_options)


class FractionOp(Operation):

    title = 'Watershed fractional area'
    name = 'fraction'
    output_type = 'fractions'
    filename_format = '{case_id}_{output_type}'

    def run(self, unique_ds, zones_ds, fine_mask_ds):
        fine_mask_array = fine_mask_ds.array
        fine_mask_array[fine_mask_array==fine_mask_ds.nodata] = 0
        fraction_np = np.zeros(unique_ds.array.shape)
        unique_iter = np.nditer(unique_ds.array, flags=['multi_index'])
        while not unique_iter.finished:
            masked = np.ma.MaskedArray(
                    fine_mask_array,
                    mask=(zones_ds.array != unique_iter[0]))
            fraction_np[unique_iter.multi_index] = masked.mean()
            unique_iter.iternext()

        # Write to gdal array
        fraction_np[fraction_np==0] = unique_ds.nodata
        fraction_ds = GDALDataset(self.path, template=unique_ds)
        fraction_ds.array = fraction_np


class GridAreaOp(Operation):

    title = 'Grid Area'
    name = 'grid-area'
    output_type = 'grid_area'
    filename_format = '{case_id}_{output_type}'

    def run(self, fraction_ds):
        fraction_ds.saveas('nc')
        subprocess.call(['cdo', 'gridarea',
                         fraction_ds.loc.nc,
                         self.path.nc])


class ClipToCoarseOp(Operation):

    title = 'Clip to outline of watershed at VIC gridcell resolution'
    output_type = 'clip_to_coarse'

    def run(self, coarse_mask_ds, fine_ds):
        # Projection
        input_srs = osr.SpatialReference(wkt=coarse_mask_ds.projection)

        # Shapefile for output (in same projection)
        shp_path = TempLocator('coarse_mask_outline', **self.cfg)
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        outline_ds = shp_driver.CreateDataSource(shp_path.shp)
        layer = 'mask_outline'
        outline_layer = outline_ds.CreateLayer(layer, srs=input_srs)
        outline_field = ogr.FieldDefn('MO', ogr.OFTInteger)
        outline_layer.CreateField(outline_field)

        # Get Band
        mask_band = coarse_mask_ds.dataset.GetRasterBand(1)

        # Polygonize
        gdal.Polygonize(mask_band, mask_band, outline_layer, -9999)
        outline_ds.SyncToDisk()
        outline_ds = None

        warp_options = gdal.WarpOptions(
                format='GTiff',
                cutlineDSName=shp_path.shp,
                dstNodata=-9999)
        gdal.Warp(self.path.gtif, fine_ds.dataset, options=warp_options)




class FlowDistanceOp(Operation):

    output_types = [OutputType('flow-distance', 'gtif')]

    def dir2dist(self, lat, lon, direction, res, nodata):
        distance = direction.astype(np.float64)
        np.copyto(distance,
                  self.distance(lon - res.lon/2, lat, lon + res.lon/2, lat),
                  where=np.isin(direction, [1,5]))
        np.copyto(distance,
                  self.distance(lon, lat - res.lat/2, lon, lat + res.lat/2),
                  where=np.isin(direction, [3,7]))
        np.copyto(distance,
                  self.distance(lon - res.lon/2, lat - res.lat/2,
                                lon + res.lon/2, lat + res.lat/2),
                  where=np.isin(direction, [2, 4, 6, 8]))
        return distance

    def run(self, flow_dir_ds):
        direction = flow_dir_ds.array
        lon, lat = np.meshgrid(flow_dir_ds.cgrid.lon, flow_dir_ds.cgrid.lat)
        res = flow_dir_ds.resolution

        direction = np.ma.masked_where(direction==flow_dir_ds.nodata,
                                       direction)
        lat = np.ma.masked_where(direction==flow_dir_ds.nodata, lat)
        lon = np.ma.masked_where(direction==flow_dir_ds.nodata, lon)

        distance = self.dir2dist(
                lat, lon, direction, res, flow_dir_ds.nodata)

        flow_dist_ds = GDALDataset(
                self.locs['flow-distance'], template=flow_dir_ds)

        # Fix masking side-effects - not sure why this needs to be done
        flow_dist_ds.nodata = -9999
        distance = distance.filled(flow_dist_ds.nodata)
        distance[distance==None] = flow_dist_ds.nodata
        distance = distance.astype(np.float32)
        flow_dist_ds.array = distance


class FlowDistanceHaversineOp(FlowDistanceOp):

    title = 'Haversine Flow Distance'
    name = 'flow-distance-haversine'

    def distance(self, lon1, lat1, lon2, lat2):
        lon1 = np.deg2rad(lon1)
        lat1 = np.deg2rad(lat1)
        lon2 = np.deg2rad(lon2)
        lat2 = np.deg2rad(lat2)
        r = 6378100 #meters
        a = np.sin((lat2 - lat1) / 2) ** 2
        b = np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a + b))


class FlowDistanceEuclideanOp(FlowDistanceOp):

    title = 'Euclidean Flow Distance'
    name = 'flow-distance-euclidean'

    def distance(self, lon1, lat1, lon2, lat2):
        return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)


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



class SoilDepthOp(Operation):
    """ Compute soil depth from slope, elevation, and source area"""

    title = 'Soil Depth'
    name = 'soil-depth'
    output_types = [OutputType('soil-depth', 'gtif')]

    def run(self, slope_ds, source_ds, elev_ds,
            min_depth, max_depth,
            wt_slope=0.7, wt_source=0.0, wt_elev=0.3,
            max_slope=30.0, max_source=100000.0, max_elev=1500.0,
            pow_slope=0.25, pow_source=1.0, pow_elev=0.75):
        wt_total = float(wt_slope + wt_source + wt_elev)

        if not wt_total == 1.0:
            logging.warning('Soil depth weights do not add up to 1.0 - scaling')
            wt_slope = wt_slope / wt_total
            wt_source = wt_source / wt_total
            wt_elev = wt_elev / wt_total

        # Scale sources: [min, max] -> [min/max, 1]
        slope_arr = np.clip(slope_ds.array/max_slope, None, 1)
        source_arr = np.clip(source_ds.array/max_source, None, 1)
        elev_arr = np.clip(elev_ds.array/max_elev, None, 1)

        # Calculate soil depth
        soil_depth_arr = min_depth + \
                (max_depth - min_depth) * (
                    wt_slope  * (1.0 - np.power(slope_arr, pow_slope)) +
                    wt_source * np.power(source_arr,       pow_source) +
                    wt_elev   * (1.0 - np.power(elev_arr,  pow_elev))
                )

        # Save in a dataset matching the DEM
        soil_depth_ds = GDALDataset(
                self.locs['soil-depth'], template=elev_ds)
        soil_depth_ds.array = soil_depth_arr


class DHSVMNetworkOp(Operation):
    """ Run the TauDEM Stream Reach and Watershed Command """

    title = 'DHSVM Network'
    name = 'dhsvm-network'
    output_types = [
        OutputType('network', 'csv'),
        OutputType('map', 'csv'),
        OutputType('state', 'csv'),
        OutputType('class', 'csv')
    ]

    net_colnames = [
        u'LINKNO',
        u'order',
        u'Slope',
        u'Length',
        u'class',
        u'DSLINKNO'
    ]
    state_colnames = [
        u'LINKNO',
        u'initial_state'
    ]
    map_colnames = [
        'xind',
        'yind',
        'channel_id',
        'efflength',
        'effdepth',
        'effwidth'
    ]
    class_colnames = [
        'class',
        'hydwidth',
        'hyddepth',
        'n',
        'inf'
    ]

    class_properties = pd.DataFrame.from_records({
        'class':    [1,    2,    3,    4,    5,    6,
                     7,    8,    9,    10,   11,   12,
                     13,   14,   15,   16,   17,   18],
        'hyddepth': [0.5,  1.0,  2.0,  3.0,  4.0,  4.5,
                     0.5,  1.0,  2.0,  3.0,  4.0,  4.5,
                     0.5,  1.0,  2.0,  3.0,  4.0,  4.5],
        'hydwidth': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                     0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
        'n':        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                     0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
        'inf':      np.zeros(18),
        'effwidth': [0.06, 0.09, 0.12, 0.15, 0.18, 0.21,
                     0.1,  0.15, 0.2,  0.25, 0.3,  0.35,
                     0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'initial_state': 0.1 * np.ones(18)
        }, index='class')

    def get_class(self, row):
        slope = row['Slope']
        mean_cont_area = row['meanContArea']

        hydclass = 1
        if slope > 0.002: hydclass += 6
        if slope > 0.01: hydclass += 6
        if mean_cont_area > 1000000: hydclass += 1
        if mean_cont_area > 10000000: hydclass += 1
        if mean_cont_area > 20000000: hydclass += 1
        if mean_cont_area > 30000000: hydclass += 1
        if mean_cont_area > 40000000: hydclass += 1

        return hydclass

    def run(self, tree_ds, coord_ds, network_ds, watershed_ds,
                soil_depth_ds, flow_distance_ds):
        coord_df = coord_ds.dataset
        tree_df = tree_ds.dataset

        ## Convert network and watershed input to dataframes
        # Network
        net_df = gpd.read_file(network_ds.loc.path)
        net_df.set_index('LINKNO')

        # Channel ID
        channelid_arr = watershed_ds.array
        nodata = -99
        channelid_arr[channelid_arr == watershed_ds.nodata] = nodata

        x, y = np.meshgrid(watershed_ds.cgrid.x, watershed_ds.cgrid.y)
        inds = np.indices(channelid_arr.shape)
        channelid_df = pd.DataFrame.from_records(
                {'x': x.flatten(),
                 'y': y.flatten(),
                 'xind': inds[1].flatten(),
                 'yind': inds[0].flatten(),
                 'effdepth': 0.01 * soil_depth_ds.array.flatten(),
                 'efflength': flow_distance_ds.array.flatten(),
                 'channel_id': channelid_arr.flatten()}
        )

        ## STREAM NETWORK FILE
        # Compute mean contributing area
        net_df['meanContArea']  = (net_df['USContArea'] +
                                   net_df['DSContArea']) / 2

        # Determine hydrologic class and related quantities
        net_df['class'] = net_df.apply(self.get_class, axis=1)
        net_df = net_df.join(self.class_properties, on='class')

        # Set a minimum value for the slope and length
        net_df.loc[net_df['Length'] <= 0.00001, 'Length'] = 0.00001
        net_df.loc[net_df['Slope'] <= 0.00001, 'Slope'] = 0.00001

        # Compute routing order
        layers = [net_df[net_df['DSLINKNO']==-1]]
        reordered = 1
        while reordered < len(net_df.index):
            upstream = (layers[-1]['USLINKNO1'].tolist() +
                        layers[-1]['USLINKNO2'].tolist())
            upstream = [a for a in upstream if a != -1]
            layers.append(net_df[net_df['LINKNO'].isin(upstream)])
            reordered += len(upstream)
        layers.reverse()
        net_df = pd.concat(layers)
        net_df['order'] = range(1, reordered + 1)

        ## STREAM MAP FILE
        # Filter out cells with no channel
        channelid_df = channelid_df[channelid_df['channel_id'] != nodata]

        # Find lat and lon ids by minimum distance
        # Comparing floats does not work!
        coord_df['id'] = coord_df.apply(
                lambda row: ((channelid_df['x'] - row['x'])**2 +
                            (channelid_df['y'] - row['y'])**2).idxmin(),
                axis='columns')

        # Join channel id, depth, and length to map dataframe
        map_df = coord_df.join(channelid_df, on='id', lsuffix='coord')

        # Add effective width from channel
        map_df = map_df.join(net_df, on='channel_id')

        # Channel IDs cannot be 0
        map_df['channel_id'] += 1
        net_df['LINKNO'] += 1
        net_df['DSLINKNO'] += 1

        ## SELECT COLUMNS AND SAVE
        # Extract stream network file columns from network shapefile
        # Must reverse order, or DHSVM will not route correctly!
        csvargs = {
            'sep': '\t',
            'float_format': '%.5f',
            'header': False,
            'index': False
        }

        logging.info('Writing network file')
        self.locs['network'].csvargs = csvargs
        net_df = net_df[::-1]
        net_df[self.net_colnames].to_csv(self.locs['network'].path, **csvargs)

        logging.info('Writing network map file')
        self.locs['map'].csvargs = csvargs
        map_df[self.map_colnames].to_csv(self.locs['map'].path, **csvargs)

        logging.info('Writing network state file')
        self.locs['state'].csvargs = csvargs
        net_df.sort_values(['order'])[self.state_colnames].to_csv(
                self.locs['state'].path, **csvargs)

        logging.info('Writing channel class file')
        class_csvargs = {
            'sep': '\t',
            'float_format': '%.2f',
            'header': False,
            'index': True
        }
        self.locs['class'].csvargs = class_csvargs
        self.class_properties.to_csv(self.locs['class'].path, **class_csvargs)


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


class UpscaleFlowDirectionOp(Operation):

    title = 'Final Resolution Flow Direction'
    name = 'upscale-flow-direction'
    output_types = ['flow-direction']

    @property
    def resolution(self):
        return self.kwargs['template_ds'].resolution

    def run(self, flow_acc_ds, template_ds):
        flowgen_path = ScriptLocator('flowgen', filename='flowgen').configure(
                                  self.cfg)
        flow_acc_path = Locator(
                filename=flow_acc_ds.loc.filename + '_nohead').configure(
                        self.cfg)
        flow_acc_ds_long = flow_acc_ds.array.astype(np.int_)

        # Flowgen seems to require an upside down array ??
        flow_acc_ds_long = np.flipud(flow_acc_ds_long)

        with open(flow_acc_path.asc, 'w') as flow_acc_file:
            flow_acc_ds_long.tofile(flow_acc_file)

        upscale_num = template_ds.resolution / flow_acc_ds.resolution

        subprocess.call([flowgen_path.path,
                         flow_acc_path.asc,
                         str(flow_acc_ds.size.y),
                         str(flow_acc_ds.size.x),
                         self.locs['flow-direction'].asc,
                         str(int(round(upscale_num.y))),
                         str(int(round(upscale_num.x))), '-v'])

        # Load ascii data into dataset with spatial reference
        flow_dir_ds = GDALDataset(self.locs['flow-direction'],
                                  template=template_ds)
        flow_dir_array = np.loadtxt(self.locs['flow-direction'].asc)

        # Flip array back
        flow_dir_array = np.flipud(flow_dir_array)

        # Adjust output values
        flow_dir_array[abs(flow_dir_array - 4.5) > 3.5] = -9999
        flow_dir_array[np.isnan(flow_dir_array)] = -9999
        flow_dir_ds.nodata = -9999
        flow_dir_ds.array = flow_dir_array


class ConvertOp(Operation):

    output_types = [OutputType('flow-direction', 'gtif')]

    def run(self, flow_dir_ds):
        converted_ds = GDALDataset(
                self.locs['flow-direction'], template=flow_dir_ds)
        convert_array = np.vectorize(self.convert)

        input_array = np.ma.masked_where(
                flow_dir_ds.array==flow_dir_ds.nodata,
                flow_dir_ds.array)
        input_array.set_fill_value(flow_dir_ds.nodata)

        converted = convert_array(input_array)
        converted_ds.array = converted.filled()


class NorthCWToEastCCWOp(ConvertOp):

    title = 'North CW (RVIC) to East CCW (TauDEM) Flow Directions'
    name = 'ncw-to-eccw'
    filename_format = 'eastccw_{output_label}'

    def convert(self, array):
        return (3 - array) % 8 + 1


class EastCCWToNorthCWOp(ConvertOp):

    title = 'East CCW (TauDEM) to North CW (RVIC) Flow Directions'
    name = 'eccw-to-ncw'
    filename_format = 'northcw_{output_label}'

    def convert(self, array):
        return (3 - array) % 8 + 1


class BasinIDOp(Operation):
    '''
    Generates an RVIC-acceptable basin ID file from a mask

    This is a placeholder to actually computing the basin ID with TauDEM
    Gage Watershed or others - it will not work with multiple basins.
    '''
    title = 'Basin ID'
    name = 'basin-id'
    output_type = 'basin_id'

    def run(self, path, mask_ds):
        basin_id_ds = GDALDataset(self.path, template=mask_ds)

        # Compute basin ID array
        basin_id_ds.array[np.where(basin_id_ds.array==1)] = 1
        basin_id_ds.array[np.where(basin_id_ds.array!=1)] = basin_id_ds.nodata


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


class NLCDtoDHSVM(Operation):
    """
    Convert vegetation classes from the National Land Cover
    Database to corresponding DHSVM values
    """

    title = 'Remap NLCD to DHSVM'
    name = 'nlcd-to-dhsvm'
    output_types = [OutputType('veg-type', 'gtif')]

    remap_table = {
        3: 12,
        11: 14,
        12: 20,
        21: 10,
        22: 13,
        23: 13,
        24: 13,
        31: 12,
        41: 4,
        42: 1,
        43: 5,
        51: 9,
        52: 9,
        71: 10,
        72: 10,
        81: 10,
        82: 11,
        90: 17,
        95: 9,
    }

    def run(self, nlcd_ds, nodata=-99):
        veg_type_ds = GDALDataset(
            self.locs['veg-type'], template=nlcd_ds)
        array = copy.copy(nlcd_ds.array)
        for nlcd, dhsvm in self.remap_table.items():
            array[nlcd_ds.array==nlcd] = dhsvm
        veg_type_ds.array = array


class SkyviewOp(Operation):
    """ Compute skyview files from a DEM """

    title = 'Skyview'
    name = 'skyview'
    output_types = [
        OutputType('skyview', 'nc')
    ]

    def run(self, elevation_ds, elevation_epsg,
            time_zone, dt=1, n_directions=8, year=2000):
        standard_meridian_srs = osr.SpatialReference()
        standard_meridian_srs.ImportFromEPSG(4326)
        elevation_srs = osr.SpatialReference()
        elevation_srs.ImportFromEPSG(int(elevation_epsg[5:]))
        dem_to_latlon = osr.CoordinateTransformation(
            elevation_srs, standard_meridian_srs)
        center = CoordProperty(
            *dem_to_latlon.TransformPoint(
                elevation_ds.center.lon, elevation_ds.center.lat)[:-1])

        # Skyview
        n_rows = str(elevation_ds.size.y)
        n_cols = str(elevation_ds.size.x)
        cell_size = str(int(elevation_ds.res.x))
        longitude = str(center.lon)
        latitude = str(center.lat)
        standard_meridian_lon = str(time_zone * 15.0)
        x_origin = str(elevation_ds.ul.x)
        y_origin = str(elevation_ds.ul.y)
        n_directions = str(n_directions)
        year = str(year)
        day = '15'
        steps_per_day = str(int(24 / dt))
        dt = str(dt)

        skyview_args = [
            self.scripts['skyview'],
            elevation_ds.loc.path, self.locs['skyview'].path,
            n_directions, n_rows, n_cols, cell_size, x_origin, y_origin
        ]
        logging.info('Calling process %s', ' '.join(skyview_args))
        skyview_process = subprocess.Popen(
            skyview_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        skyview_output, _ = skyview_process.communicate()
        logging.info(skyview_output)


class ShadowOp(Operation):
    """ Compute shadow and skyview files from a DEM """

    title = 'Monthly Shadow'
    name = 'shadow'
    output_types = [
        OutputType('shadow-hourly', 'nc'),
        OutputType('shadow', 'nc')
    ]

    def run(self, elevation_ds, elevation_epsg,
            time_zone, dt=1, n_directions=8, year=2000):
        standard_meridian_srs = osr.SpatialReference()
        standard_meridian_srs.ImportFromEPSG(4326)
        elevation_srs = osr.SpatialReference()
        elevation_srs.ImportFromEPSG(int(elevation_epsg[5:]))
        dem_to_latlon = osr.CoordinateTransformation(
            elevation_srs, standard_meridian_srs)
        center = CoordProperty(
            *dem_to_latlon.TransformPoint(
                elevation_ds.center.lon, elevation_ds.center.lat)[:-1])

        # Skyview
        n_rows = str(elevation_ds.size.y)
        n_cols = str(elevation_ds.size.x)
        cell_size = str(int(elevation_ds.res.x))
        longitude = str(center.lon)
        latitude = str(center.lat)
        standard_meridian_lon = str(time_zone * 15.0)
        x_origin = str(elevation_ds.ul.x)
        y_origin = str(elevation_ds.ul.y)
        n_directions = str(n_directions)
        year = str(year)
        day = '15'
        steps_per_day = str(int(24 / dt))
        dt = str(dt)

        for (hourly_loc, _), (monthly_loc, meta) in zip(
                self.locs['shadow-hourly'], self.locs['shadow']):
            month = str(meta['month'])
            # Hourly shadow
            shading_args = [
                self.scripts['shading'],
                elevation_ds.loc.path, hourly_loc.path,
                n_rows, n_cols, cell_size,
                longitude, latitude, standard_meridian_lon,
                year, month, day, dt,
                x_origin, y_origin
            ]
            logging.info('Calling process %s', ' '.join(shading_args))
            shading_process = subprocess.Popen(
                shading_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            shading_output, _ = shading_process.communicate()
            logging.info(shading_output)

            # Monthly average
            average_args = [
                self.scripts['average'],
                hourly_loc.path, monthly_loc.path,
                '24', steps_per_day, n_rows, n_cols, cell_size,
                x_origin, y_origin, month
            ]
            logging.info('Calling process %s', ' '.join(average_args))
            average_process = subprocess.Popen(
                average_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            average_output, _ = average_process.communicate()
            logging.info(average_output)


class AverageLayers(Operation):
    """ Weighted average of layers by depth """

    title = "Average Layers"
    name = 'average-layers'
    output_types = [OutputType('average', 'gtif')]

    def run(self, layered_ds):
        diff = []
        for layer in layered_ds:
            diff.append(layer.meta['layer_max'] - layer.meta['layer_min'])
        weights = [d / sum(diff) for d in diff]

        average = 0 * np.ones(layered_ds[0].array.shape)
        for layer, weight in zip(layered_ds, weights):
            average += layer.array * weight

        average_ds = GDALDataset(self.locs['average'], template=layered_ds[0])
        average_ds.array = average


class SoilType(Operation):
    """ Soil texture category from percent clay/sand/slit """

    title = "Soil Type"
    name = 'soil-type'
    output_types = [OutputType('type', 'gtif')]

    textures = {
        'Sand': 1,
        'Loamy sand': 2,
        'Sandy loam': 3,
        'Silty loam': 4,
        'Silt': 5,
        'Loam': 6,
        'Sandy clay loam': 7,
        'Silty clay loam': 8,
        'Clay loam': 9,
        'Sandy clay': 10,
        'Silty clay': 11,
        'Clay': 12,
        'Loamy sand': 13,
    }

    def run(self, texture_ds):
        types = {ds.meta['type']: ds for ds in texture_ds}
        clay = types['clay'].array
        sand = types['sand'].array
        silt = types['silt'].array
        t = 0 * np.ones(clay.shape)

        t = np.where(
            (clay > 40) & (silt > 40), self.textures['Silty clay'], t)
        t = np.where(
            (t==0) & (clay > 40) & (sand < 56),
                self.textures['Clay'], t)
        t = np.where(
            (t==0) & (clay > 28) & (sand < 20),
            self.textures['Silty clay loam'], t)
        t = np.where(
            (t==0) & (clay > 28) & (sand < 44),
            self.textures['Silty clay loam'], t)
        t = np.where(
            (t==0) & (clay > 36), self.textures['Sandy clay'], t)
        t = np.where(
            (t==0) & (clay > 20) & (silt < 28), self.textures['Sandy clay'], t)
        t = np.where(
            (t==0) & (clay < 12) & (silt > 80), self.textures['Silt'], t)
        t = np.where(
            (t==0) & (silt > 50), self.textures['Silty loam'], t)
        t = np.where(
            (t==0) & (clay > 8) & (sand < 52), self.textures['Loam'], t)
        t = np.where(
            (t==0) & (sand - clay < 70), self.textures['Sandy loam'], t)
        t = np.where(
            (t==0) & (sand - clay / 4. < 87.5), self.textures['Loamy sand'], t)
        t = np.where(t==0, self.textures['Sand'], t)

        texture_ds = GDALDataset(self.locs['type'], template=types['clay'])
        texture_ds.array = t


class DHSVMStationOp(Operation):
    """ Extract station data from NLDAS """

    title = 'DHSVM Station File'
    name = 'dhsvm-station'
    output_types = [OutputType('station', 'tsv')]
    columns = [
        'air_temp',
        'wind_speed',
        'relative_humidity',
        'incoming_shortwave',
        'incoming_longwave',
        'precipitation'
    ]
    vars = {
        'air_temp': 'tmp2m',
        'wind_speed_u': 'ugrd10m',
        'wind_speed_v': 'vgrd10m',
        'specific_humidity': 'spfh2m',
        'pressure': 'pressfc',
        'incoming_shortwave': 'dswrfsfc',
        'incoming_longwave': 'dlwrfsfc',
        'precipitation': 'apcpsfc'
    }
    cfg_template = """
        Station Name {i}     = Station{i}
        North Coordinate {i} = {y}
        East Coordinate {i}  = {x}
        Elevation {i}        = {elevation}
        Station File {i}     = input/station/Station{i}.tsv
    """


    def run(self, start, end, dt, time_zone,
            nldas_ds, elevation_ds, projected_epsg, precip_adj=1):
        station_ds = []

        time = nldas_ds.dataset.variables['time']
        start_i = nc4.date2index(start, time) - time_zone
        end_i = nc4.date2index(end, time) - time_zone
        datetimes = pd.date_range(start, end, freq=dt, closed='left')

        records = []
        j = 1
        logging.debug(nldas_ds.dataset[nldas_ds.xdim][:])
        logging.debug(nldas_ds.dataset[nldas_ds.ydim][:])
        for i, coord in enumerate(nldas_ds.coords):
            x = np.floor(
                (coord.x - nldas_ds.cmin.x) / nldas_ds.res.x
                ).astype(int)
            y = np.floor(
                (coord.y - nldas_ds.cmin.y) / nldas_ds.res.y
                ).astype(int)

            nldas_srs = osr.SpatialReference()
            nldas_srs.ImportFromEPSG(4326)
            proj_srs = osr.SpatialReference()
            proj_srs.ImportFromEPSG(int(projected_epsg[5:]))
            nldas_to_proj = osr.CoordinateTransformation(
                nldas_srs, proj_srs)

            # TransformPoint reverses coordinate order
            station_proj = CoordProperty(
                *nldas_to_proj.TransformPoint(
                    coord.y, coord.x)[:-1])
            logging.debug(station_proj.x)
            logging.debug(station_proj.y)
            logging.debug(elevation_ds.bbox)
            logging.debug(elevation_ds.bbox.contains(station_proj))
            if elevation_ds.bbox.contains(station_proj):
                record = {'i': j,
                          'xi': x, 'yi': y,
                          'x': station_proj.x, 'y': station_proj.y,
                          'elevation': elevation_ds.get_value(station_proj)}
                logging.debug(record)
                records.append(record)
                j += 1

        csvargs = {
            'sep': '\t',
            'float_format': '%.5f',
            'header': False,
            'index_label': 'datetime',
            'date_format': '%m/%d/%Y-%H',
            'columns': self.columns
        }

        self.locs['station'] = ListLoc(
            template=self.locs['station'],
            #id={'xi': 'xi', 'yi': 'yi'},
            meta=records
        ).configure(self.cfg)

        for loc, meta in self.locs['station']:
            ds = DataFrameDataset(loc).new(
                index = datetimes,
                columns = [
                    'air_temp',
                    'wind_speed_u',
                    'wind_speed_v',
                    'wind_speed',
                    'specific_humidity',
                    'pressure',
                    'vapor_pressure',
                    'sat_vapor_pressure',
                    'relative_humidity',
                    'incoming_shortwave',
                    'incoming_longwave',
                    'precipitation'
                ]
            )

            ds.csvargs = csvargs
            df = ds.dataset
            for name, id in self.vars.items():
                slice = []
                for dim in nldas_ds.dataset[id].dimensions:
                    if dim == 'time':
                        slice.append('{start_i}:{end_i}'.format(
                            start_i=start_i, end_i=end_i))
                    elif dim == nldas_ds.xdim:
                        slice.append(str(meta['xi']))
                    elif dim == nldas_ds.ydim:
                        slice.append(str(meta['yi']))
                expr = "nldas_ds.dataset['{id}'][{slice}]".format(
                    id=id, slice=','.join(slice))
                logging.debug(expr)
                col = eval(expr)
                df[name] = eval(expr)

            # mm/hour to m/hour, adjusted for calibration
            df['precipitation'] = df['precipitation'] / 1000. * precip_adj
            df['wind_speed'] = np.sqrt(
                df['wind_speed_u']**2 + df['wind_speed_v']**2 )
            df['air_temp'] = df['air_temp'] - 273.15
            df['relative_humidity'] = (
                df.apply(
                    lambda row:
                        (100 * relative_humidity_from_specific_humidity(
                            row['specific_humidity'],
                            row['air_temp'] * units.degC,
                            row['pressure'] * units.pascal).magnitude),
                    axis=1 ))
            df.loc[df.relative_humidity >= 100, 'relative_humidity'] = 100
            ds.save()


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


class DEMForDHSVM(WriteOneVarNetCDFOp):

    name = 'dhsvm-dem'
    varname = 'Basin.DEM'
    dtype = 'f'
    units = 'Meters'
    fill = -999.

class MaskForDHSVM(WriteOneVarNetCDFOp):

    name = 'dhsvm-mask'
    varname = 'Basin.Mask'
    dtype = 'i1'
    units = 'mask'
    fill = 0

class SoilDepthForDHSVM(WriteOneVarNetCDFOp):

    name = 'dhsvm-soil-depth'
    varname = 'Soil.Depth'
    dtype = 'f'
    units = 'Meters'
    fill = -999.

class VegTypeForDHSVM(WriteOneVarNetCDFOp):

    name = 'dhsvm-veg-type'
    varname = 'Veg.Type'
    dtype = 'i1'
    units = 'Veg Type'
    fill = -99

class SoilTypeForDHSVM(WriteOneVarNetCDFOp):

    name = 'dhsvm-soil-type'
    varname = 'Soil.Type'
    dtype = 'i1'
    units = 'Soil Type'
    fill = -99

class PRISMForDHSVM(WriteOneVarNetCDFOp):

    name = 'dhsvm-prism'
    varname = 'PRISM.Precip'
    dtype = 'f'
    units = 'mm*100'
    fill = -99


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

class InterceptionStateForDHSVM(WriteUniformNetCDFOp):

    name = 'dhsvm-interception-state'
    varnames = [
        '{layer}.Precip.IntRain',
        '{layer}.Precip.IntSnow',
        'Temp.Instor']
    layers = [2, 2, 1]
    dtypes = ['f', 'f', 'f']
    units = ['m', 'm', 'm']
    fills = [0, 0, 0]

class SnowStateForDHSVM(WriteUniformNetCDFOp):

    name = 'dhsvm-snow-state'
    varnames = [
        'Snow.HasSnow',
        'Snow.LastSnow',
        'Snow.Swq',
        'Snow.PackWater',
        'Snow.TPack',
        'Snow.SurfWater',
        'Snow.TSurf',
        'Snow.ColdContent']
    layers = [1, 1, 1, 1, 1, 1, 1, 1]
    dtypes = ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f']
    units = ['boolean', 'days', 'm', 'm', 'degrees C', 'm', 'degrees C', 'J']
    fills = [0, 200, 0, 0, 0, 0, 0, 0]

class SoilStateForDHSVM(WriteUniformNetCDFOp):

    name = 'dhsvm-soil-state'
    varnames = [
        '{layer}.Soil.Moist',
        'Soil.TSurf',
        '{layer}.Soil.Temp',
        'Soil.Qst',
        'Soil.Runoff']
    layers = [4, 1, 4, 1, 1]
    dtypes = ['f', 'f', 'f', 'f', 'f']
    units = ['fraction', 'degrees C', 'degrees C', 'J', 'm']
    fills = [0.1, 0, 0, 0, 0]
