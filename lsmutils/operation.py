import copy
import csv
import gdal
import geopandas as gpd
import glob
import logging
import numpy as np
import ogr
import os
import osr
import pandas as pd
import re
import shutil
import subprocess
import yaml

from path import Path, ScriptPath
from dataset import GDALDataset, BoundaryDataset
from utils import CoordProperty

class OperationMeta(type):
    
    def __children__(cls):
        children = {}
        for child in cls.__subclasses__():
            children.update({child.op_id: child})
            children.update(child.__children__())
        return children


class Operation(object):
    """
    A parent class to perform some operation on a raster dataset

    Child classes must define at minimum:
      - name: a human-readable name, 
      - type_id: a path-friendly identifier
      - run(): a function to perform the raster operation.
    """

    __metaclass__ = OperationMeta
    
    name = NotImplemented
    op_id = NotImplemented
    output_types = NotImplemented
    filename_format = '{seq_id}_{output_type}'
    working_ext = 'gtif'

    start_msg = 'Calculating {name}'
    end_msg = '{name} saved at {path}'
    error_msg = '{name} calculation FAILED'
    
    def __init__(self, run_config, seq_id, run_id='',
                 plot_boundary=None, paths=[], **kwargs):

        logging.info(self.op_id)
        # Extract configuration variables
        self.kwargs = kwargs
        self.run_config = run_config
        self.log_level = run_config['log_level']

        # Plot settings
        self.plot = self.kwargs.pop('plot', False)
        self.quiver = self.kwargs.pop('quiver', False)
        self.colorlog = self.kwargs.pop('colorlog', False)
        if 'boundary_ds' in self.kwargs:
            self.boundary_ds = self.kwargs['boundary_ds']
        elif not plot_boundary is None:
            self.boundary_ds = plot_boundary
        else:
            self.boundary_ds = None

        # IDs
        self.case_id = run_config['case_id']
        self.base_dir = run_config['base_dir']
        if run_id:
            self.filename_format = '{run_id}_' + self.filename_format
        
        # Get non-automatic attributes for filename
        self.attributes = kwargs.copy()
        self.attributes['res'] = self.resolution
        self.attributes['seq_id'] = seq_id
        self.attributes['run_id'] = run_id
        self.attributes.update(
                {name: getattr(self, name) for name in dir(self)})
        
        self.filenames = {
            ot: self.filename_format.format(
                    output_type=ot, **self.attributes).replace('.', '-')
            for ot in self.output_types}
        if paths:
            self.paths = paths
        else:
            self.paths = {
                ot: Path(filename=fn).configure(run_config)
                for ot, fn in self.filenames.items()}

        # Initialize variables
        self.datasets = []

    @property
    def input_datasets(self):
        return [ds for key, ds in self.kwargs.items()
                    if key.endswith('_ds')]
    
    @property
    def resolution(self):
        if self.input_datasets:
            for ds in self.input_datasets:
                try:
                    return ds.resolution
                except AttributeError:
                    continue
        if 'resolution' in self.attributes:
            return CoordProperty(self.attributes['resolution'],
                                 self.attributes['resolution'])
        return CoordProperty(0, 0)
        
    def run(self, **kwargs):
        raise NotImplementedError
        
    def save(self):
        """A convenience function to save as default format"""
        return self.saveas('gtif')

    def saveas(self, out_ext, datatype=None):
        print(self.start_msg.format(name=self.name))
        
        # Change path extensions to match working extension
        for ot, pth in self.paths.items():
            pth.default_ext = self.working_ext

        # Perform raster operation
        if not self.datasets:
            if (all([os.path.exists(pth.ext(out_ext)) 
                        for ot, pth in self.paths.items()])
                    and not self.log_level==logging.DEBUG):
                self.datasets = {ot: GDALDataset(pth) 
                                    for ot, pth in self.paths.items()}
            else:
                self.datasets = self.run(**self.kwargs)

        # Convert to desired file format
        if out_ext != self.working_ext or not datatype is None:
            for ot, ds in self.datasets.items():
                ds.saveas(out_ext, datatype=datatype)

        # Plot results
        if self.plot:
            for ds in self.datasets:
            	ds.plot(self.name, self.path.png,
                	boundary_ds=self.boundary_ds,
                        quiver=self.quiver, colorlog=self.colorlog)

        if self.datasets:
            out_paths = {ot: pth.ext(out_ext)
                             for ot, pth in self.paths.items()}
            print(self.end_msg.format(name=self.name, path=out_paths))
        else:
            logging.error(self.error_msg.format(name=self.name))

        return self.datasets

class MergeOp(Operation):

    name = 'Merged Raster'
    op_id = 'merge'
    output_types = ['merged']

    def run(self, raster_list):
        subprocess.call(['gdal_merge.py', '-o', self.paths['merged'].path] +
                        [ds.filepath.path for ds in raster_list])
        return {'merged': GDALDataset(self.paths['merged'])}

class AggregateOp(Operation):

    name = 'Aggregate'
    op_id = 'agg'
    output_types = ['aggregated']

    def run(self, input_ds, boundary_ds=None, 
            resolution=CoordProperty(x=1/240., y=1/240.), grid_res=None,
            padding=CoordProperty(x=0, y=0), bbox=None, algorithm='bilinear'):
        if grid_res is None:
            grid_res = resolution
        if not boundary_ds is None:
            grid_box = boundary_ds.gridcorners(
                    grid_res, padding=padding).warp_output_bounds
        elif not bbox is None:
            grid_box = [bbox[0] - padding.x, bbox[1] - padding.x,
                        bbox[2] + padding.y, bbox[3] + padding.y]
        else:
            grid_box = input_ds.gridcorners(
                    grid_res, padding=padding).warp_output_bounds
        agg_warp_options = gdal.WarpOptions(
                xRes = resolution.x,
                yRes = resolution.y,
                outputBounds = grid_box,
                targetAlignedPixels = True,
                resampleAlg=algorithm,
        )
        gdal.Warp(self.paths['aggregated'].path, input_ds.dataset, 
                  options=agg_warp_options)
        return {'aggregated': GDALDataset(self.paths['aggregated'])}
    
class ClipOp(Operation):

    name = 'Raster clipped to boundary'
    op_id = 'clip'
    output_type = 'clipped_raster'
    
    def run(self, path, input_ds, boundary_ds):
        clip_warp_options = gdal.WarpOptions(
                format='GTiff',
                cutlineDSName=boundary_ds.filepath.shp,
                cutlineBlend=.5,
                dstNodata=input_ds.nodata,
                )
        gdal.Warp(path, input_ds.dataset, options=clip_warp_options)
        return GDALDataset(self.path)

class ReProjectOp(Operation):

    name = 'Reproject raster'
    op_id = 'reproject'
    output_types = ['reprojected']

    def run(self, input_ds, proj):
        agg_warp_options = gdal.WarpOptions(dstSRS = proj)
        gdal.Warp(self.paths['reprojected'].path, input_ds.dataset, 
                  options=agg_warp_options)
        return {'reprojected': GDALDataset(self.paths['reprojected'])}

class SumOp(Operation):

    name = 'Sum raster'
    op_id = 'sum'
    output_type = 'csv'

    def run(self, path, input_ds, label_ds=None, fraction_ds=None):
        array = np.ma.masked_where(input_ds.array==input_ds.nodata,
                                   input_ds.array)
        
        if not fraction_ds is None:
            fractions = np.ma.masked_where(input_ds.array==input_ds.nodata,
                                           fraction_ds.array)
            array = array * fractions
            
        sum = array.sum()

        if label_ds is None:
            label = input_ds.filepath.filename
        else:
            label = label_ds.filepath.filename
            
        with open(self.path.csv, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([label, sum])
        return sum

    
class MaskOp(Operation):

    name = 'Raster mask of boundary area'
    op_id = 'mask'

    def run(self, path, input_ds, boundary_ds):
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
        return output_ds

class UniqueOp(Operation):

    name = 'A unique number in each VIC-resolution grid cell'
    op_id = 'unique'
    output_type = 'unique'
    filename_format = '{case_id}_{output_type}'
    
    def run(self, path, input_ds):
        # Generate unique dataset
        unique_ds = GDALDataset(self.path, template=input_ds)
        unique_ds.array = np.arange(
                1,
                unique_ds.array.size + 1
        ).reshape(unique_ds.array.shape)
        return unique_ds

class ZonesOp(Operation):

    name = 'Downscaled Zones for each VIC gridcell'
    op_id = 'zones'
    output_type = 'zones'
    filename_format = '{case_id}_{output_type}'
    
    def run(self, path, unique_ds, fine_ds):
        res_warp_options = gdal.WarpOptions(
                xRes = fine_ds.res.x,
                yRes = fine_ds.res.y,
                outputBounds = fine_ds.rev_warp_output_bounds,
                resampleAlg='average')
        gdal.Warp(path, unique_ds.dataset, options=res_warp_options)
        return GDALDataset(self.path)

class FractionOp(Operation):

    name = 'Watershed fractional area'
    op_id = 'fraction'
    output_type = 'fractions'
    filename_format = '{case_id}_{output_type}'

    def run(self, path, unique_ds, zones_ds, fine_mask_ds):
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
        return fraction_ds

class GridAreaOp(Operation):

    name = 'Grid Area'
    op_id = 'grid-area'
    output_type = 'grid_area'
    filename_format = '{case_id}_{output_type}'

    def run(self, path, fraction_ds):
        fraction_ds.saveas('nc')
        subprocess.call(['cdo', 'gridarea',
                         fraction_ds.filepath.nc,
                         self.path.nc])
        return GDALDataset(self.path, filetype='nc')

class ClipToCoarseOp(Operation):

    name = 'Clip to outline of watershed at VIC gridcell resolution'
    output_type = 'clip_to_coarse'

    def run(self, path, coarse_mask_ds, fine_ds):
        # Projection
        input_srs = osr.SpatialReference(wkt=coarse_mask_ds.projection)
        
        # Shapefile for output (in same projection)
        shp_path = TempPath('coarse_mask_outline', **self.run_config)
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
        
        return GDALDataset(self.path)

class RemoveSinksOp(Operation):

    name = 'Sinks Removed'
    op_id = 'remove-sinks'
    output_types = ['no-sinks']
    working_ext = 'tif'

    def run(self, input_ds):
        # Remove Pits / Fill sinks
        subprocess.call(['pitremove',
                         '-z', input_ds.filepath.path,
                         '-fel', self.paths['no-sinks'].no_ext])
        no_sinks_ds = GDALDataset(self.paths['no-sinks'], filetype='tif')
        return {'no-sinks': no_sinks_ds}
        
class FlowDirectionOp(Operation):

    name = 'Flow Direction'
    op_id = 'flow-direction'
    output_types = ['flow-direction']
    working_ext = 'tif'

    def run(self, dem_ds):
        # Calculate Flow Directions
        subprocess.call(['d8flowdir', '-fel', dem_ds.filepath.tif, '-p',
                         self.paths['flow-direction'].no_ext])
        flow_dir_ds = GDALDataset(self.paths['flow-direction'], filetype='tif')
        return {'flow-direction': flow_dir_ds}

class FlowDistanceOp(Operation):

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
        
    def run(self, path, flow_dir_ds):
        direction = flow_dir_ds.array
        lon, lat = np.meshgrid(flow_dir_ds.cgrid.lon, flow_dir_ds.cgrid.lat)
        res = flow_dir_ds.resolution

        direction = np.ma.masked_where(direction==flow_dir_ds.nodata, direction)
        lat = np.ma.masked_where(direction==flow_dir_ds.nodata, lat)
        lon = np.ma.masked_where(direction==flow_dir_ds.nodata, lon)
        
        #dist_func = np.vectorize(self.dir2dist, excluded=['res', 'nodata'])
        distance = self.dir2dist(lat, lon, direction, res, flow_dir_ds.nodata)

        output_ds = GDALDataset(self.path, template=flow_dir_ds)
        
        # Fix masking side-effects - not sure why this needs to be done
        output_ds.nodata = -9999
        distance = distance.filled(output_ds.nodata)
        distance[distance==None] = output_ds.nodata
        distance = distance.astype(np.float32)
        output_ds.array = distance
        return output_ds


class FlowDistanceHaversineOp(FlowDistanceOp):

    name = 'Haversine Flow Distance'
    op_id = 'flow-distance-haversine'
    output_type = 'flow_dist'

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

    name = 'Euclidean Flow Distance'
    op_id = 'flow-distance-euclidean'
    output_type = 'flow_dist'

    def distance(self, lon1, lat1, lon2, lat2):
        return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
    
class FlowAccumulationOp(Operation):

    name = 'Flow Accumulation'
    op_id = 'flow-accumulation'
    output_types = ['flow-accumulation']
    working_ext = 'tif'

    def run(self, flow_dir_ds):
        flow_dir_path = flow_dir_ds.filepath.ext(flow_dir_ds.filetype)
        subprocess.call(['aread8',
                         '-p', flow_dir_path,
                         '-ad8', self.paths['flow-accumulation'].no_ext,
                         '-nc'])
        output_ds = GDALDataset(self.paths['flow-accumulation'], filetype='tif')
        output_ds.nodata = 0
        return {'flow-accumulation': output_ds}

    
class StreamDefinitionByThresholdOp(Operation):
    """ Run the TauDEM Stream Definition By Threshold Command """
    
    name = 'Stream Definition By Threshold'
    op_id = 'stream-definition-threshold'
    output_type = 'stream-raster'
    requires = ['flow_accum']

    def run(self, path, flow_acc_ds):
        threshold = np.percentile(flow_acc_ds.array, 98)
        subprocess.call(['threshold',
                         '-ssa', flow_acc_ds.filepath.tif,
                         '-thresh', '{:.1f}'.format(threshold),
                         '-src', self.path.no_ext])
        out_ds = GDALDataset(self.path, filetype='tif')
        return out_ds

    
class MoveOutletsToStreamOp(Operation):
    """ Run the TauDEM Move Outlets to Streams Command """
    
    name = 'Move Outlets to Streams'
    op_id = 'snap-outlet'
    output_type = 'outlet-on-stream'
    requires = ['flow_dir', 'stream_raster_thresh', 'outlet']

    def run(self, path, flow_dir_ds, stream_ds, outlet_ds):
        tmp = TempPath('snapped', **self.run_config)
        subprocess.call(['moveoutletstostrm',
                         '-p', flow_dir_ds.filepath.tif,
                         '-src', stream_ds.filepath.tif,
                         '-o', outlet_ds.filepath.shp,
                         '-om', tmp.shp])
        
        # Copy SRS from outlet
        in_ds = BoundaryDataset(tmp)
        # This should not be necessary - shp should be the default extension
        out_ds = BoundaryDataset(self.path, update=True).new()

        # Create layer with outlet srs
        in_lyr = in_ds.dataset.GetLayer()
        outlet_lyr = outlet_ds.dataset.GetLayer()
        srs = outlet_lyr.GetSpatialRef()
        geom_type = in_lyr.GetLayerDefn().GetGeomType()
        out_lyr = out_ds.dataset.CreateLayer('outlet', srs, geom_type)
        
        # Copy outlet in correct srs
        outlet = in_lyr.GetFeature(0)
        out_feature = ogr.Feature(out_lyr.GetLayerDefn())
        out_feature.SetGeometry(outlet.GetGeometryRef().Clone())
        out_lyr.CreateFeature(out_feature)

        # Clean up
        out_ds.dataset.Destroy()
        in_ds.dataset.Destroy()
        out_ds = BoundaryDataset(self.path)
        return out_ds

class LabelGagesOp(Operation):
    """ Add a sequential id field to each outlet in a shapefile """

    name = 'Labeled Gages'
    op_id = 'label-outlet'
    output_type = 'labelled-outlet'
    requires = ['snap_outlet']

    def run(self, path, outlet_ds):
        outlet_ds.chmod(True)
        outlet_path = outlet_ds.filepath
        
        layer = outlet_ds.dataset.GetLayer()
        id_field = ogr.FieldDefn('id', ogr.OFTInteger)
        layer.CreateField(id_field)

        feature = layer.GetNextFeature()
        gage_id = 1

        while feature:
            feature.SetField("id", gage_id)
            layer.SetFeature(feature)
            feature = layer.GetNextFeature()
            gage_id += 1

        # Clean up and copy to correct location
        outlet_ds.dataset.Destroy()
        for a_file in glob.glob(r'{}.*'.format(outlet_path.no_ext)):
            shutil.copyfile(a_file,
                            self.path.ext(os.path.splitext(a_file)[1][1:]))
        outlet_ds = BoundaryDataset(self.path)
        return outlet_ds
            

class GageWatershedOp(Operation):
    """ 
    Run the TauDEM Gage Watershed Command 

    Raster labeling each point by which gage it drains to directly
    """
    
    name = 'Gage Watershed'
    op_id = 'gage-watershed'
    output_type = 'boundary-raster'
    requires = ['flow_dir', 'snap_outlet']

    def run(self, path, flow_dir_ds, outlet_ds):
        subprocess.call(['gagewatershed',
                         '-p', flow_dir_ds.filepath.tif,
                         '-o', outlet_ds.filepath.shp,
                         '-gw', self.path.tif])
        out_ds = GDALDataset(self.path, filetype='tif')
        return out_ds

    
class PeukerDouglasStreamDefinitionOp(Operation):
    """ Run the TauDEM Peuker Douglas Stream Definition Command """
    
    name = 'Peuker Douglas Stream Definition'
    op_id = 'stream-def-pd'
    output_type = 'stream_definition'
    requires = ['no_sinks', 'flow_dir', 'outlets', 'flow_accum']

    def run(self, path, no_sinks_ds, flow_dir_ds, flow_accum_ds, outlet_ds):
        # The threshold range should be selected based on the raster size
        # Something like 10th to 99th percentile of flow accumulation?
        pd_path = TempPath('dropanalysis', **self.run_config)
        ssa_path = TempPath('ssa', **self.run_config)

        ## This is a three-step process - first compute the stream source grid
        subprocess.call(['aread8',
                         '-p', flow_dir_ds.filepath.tif,
                         '-o', outlet_ds.filepath.shp,
                         '-ad8', ssa_path.tif])

        ## Next perform the drop analysis
        # This selects a sensible flow accumulation threshold value
        subprocess.call(['dropanalysis',
                         '-p', flow_dir_ds.filepath.tif,
                         '-fel', no_sinks_ds.filepath.tif,
                         '-ad8', flow_accum_ds.filepath.tif,
                         '-o', outlet_ds.filepath.shp,
                         '-ssa', ssa_path.tif,
                         '-drp', pd_path.txt,
                         '-par', '5', '2000', '20', '1'])

        ## Finally define the stream
        # Extract the threshold from the first row with drop statistic t < 2
        with open(pd_path.txt, 'r') as drop_file:
            # Get optimum threshold value from last line
            for line in drop_file:
                pass
            last = line
            thresh_re = re.compile(r'([\.\d]*)$')
            threshold = thresh_re.search(last).group(1)

        subprocess.call(['threshold',
                         '-ssa', ssa_path.tif,
                         '-thresh', threshold,
                         '-src', self.path.tif])

        out_ds = GDALDataset(self.path, filetype='tif')
        return out_ds

    
class StreamNetworkOp(Operation):
    """ Run the TauDEM Stream Reach and Watershed Command """
    
    name = 'Stream Network'
    op_id = 'stream-network'
    output_type = 'stream_network'

    tree_colnames = [
        'link_no',
        'start_coord',
        'end_coord',
        'next_link',
        'previous_link',
        'order',
        'monitoring_point_id',
        'network_magnitude'
    ]
    coord_colnames = [
        'x',
        'y',
        'distance_to_terminus',
        'elevation',
        'contributing_area'
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
        'yind',
        'xind',
        'channel_id',
        'efflength',
        'effdepth',
        'effwidth'
    ]
    
    class_properties = pd.DataFrame.from_records({
        'class':    [1,   2,   3,   4,   5,   6,
                     7,   8,   9,   10,  11,  12,
                     13,  14,  15,  16,  17,  18],
        'hyddepth': [0.5, 1.0, 2.0, 3.0, 4.0, 4.5,
                     0.5, 1.0, 2.0, 3.0, 4.0, 4.5,
                     0.5, 1.0, 2.0, 3.0, 4.0, 4.5],
        'hydwidth': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                     0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
        'effwidth': [0.06, 0.09, 0.12, 0.15, 0.18, 0.21,
                     0.1,  0.15, 0.2,  0.25, 0.3,  0.35,
                     0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
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
    
    def run(self, path, no_sinks_ds, flow_dir_ds, flow_accum_ds,
            pd_stream_def_ds, outlet_ds, flow_distance_ds, soil_depth_ds):

        order_path = TempPath('{}.order'.format(self.path.filename),
                              default_ext='tif', **self.run_config)
        tree_path = TempPath('{}.tree'.format(self.path.filename),
                             default_ext='dat', **self.run_config)
        coord_path = TempPath('{}.coord'.format(self.path.filename),
                              default_ext='dat', **self.run_config)
        wshd_path = TempPath('{}.wshd'.format(self.path.filename),
                              default_ext='tif', **self.run_config)
        state_path = TempPath('{}.state'.format(self.path.filename),
                              default_ext='dat', **self.run_config)
        
        subprocess.call(['streamnet',
                         '-p', flow_dir_ds.filepath.tif,
                         '-fel', no_sinks_ds.filepath.tif,
                         '-ad8', flow_accum_ds.filepath.tif,
                         '-src', pd_stream_def_ds.filepath.tif,
                         '-o', outlet_ds.filepath.shp,
                         '-ord', order_path.path,
                         '-tree', tree_path.path,
                         '-coord', coord_path.path,
                         '-net', self.path.shp,
                         '-w', wshd_path.path])

        # Get all the output information from the stream network as dataframes
        tree_df = pd.read_csv(tree_path.path, delimiter='\t',
                              header=None, names=self.tree_colnames)
        
        net_df = gpd.read_file(self.path.shp)
        net_df.set_index('LINKNO')
        
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
        # Open datasets
        coord_df = pd.read_csv(coord_path.path, delimiter='\t',
                               header=None, names = self.coord_colnames)
        
        channelid_ds = GDALDataset(wshd_path, filetype='tif')
        channelid_arr = channelid_ds.array
        nodata = -99
        channelid_arr[channelid_arr == channelid_ds.nodata] = nodata

        # Convert rasters to dataframe
        x, y = np.meshgrid(channelid_ds.cgrid.x, channelid_ds.cgrid.y)
        inds = np.indices(channelid_arr.shape)
        channelid_df = pd.DataFrame.from_records(
                {'x': x.flatten(),
                 'y': y.flatten(),
                 'xind': inds[0].flatten(),
                 'yind': inds[1].flatten(),
                 'effdepth': 0.95 * soil_depth_ds.array.flatten(),
                 'efflength': flow_distance_ds.array.flatten(),
                 'channel_id': channelid_arr.flatten()}
        )
        
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
        net_df = net_df[::-1]
        net_df[self.net_colnames].to_csv(
                self.path.ext('network.dat'), sep='\t',
                float_format='%.5f', header=False, index=False)
        
        map_df[self.map_colnames].to_csv(
                self.path.ext('map.dat'), sep='\t', float_format='%.5f',
                header=None, index=False)

        net_df[self.state_colnames].sort(['order']).to_csv(
                state_path.path, sep='\t',
                float_format='%.5f', header=False, index=False)
                
        out_ds = GDALDataset(wshd_path)
        return out_ds

    
class DinfFlowDirOp(Operation):
    """ Compute Slope and Aspect from a DEM """

    name = 'D-infinity Flow Direction'
    op_id = 'dinf-flow-direction'
    output_types = ['slope', 'aspect']
    working_ext = 'tif'

    def run(self, elevation_ds):
        subprocess.call(['dinfflowdir',
                         '-fel', elevation_ds.filepath.path,
                         '-slp', self.paths['slope'].tif,
                         '-ang', self.paths['aspect'].tif])
        slope_ds = GDALDataset(self.paths['slope'], filetype='tif')
        aspect_ds = GDALDataset(self.paths['aspect'], filetype='tif')
        return {'slope': slope_ds, 'aspect': aspect_ds}

class Slope(Operation):
    """ 
    Compute Slope from Elevation 
    
    This operation REQUIRES a projected coordinate system in the same
    units as elevation!
    """

    name = 'Slope'
    op_id = 'slope'
    output_types = ['slope']

    def run(self, elevation_ds):
        # Take the gradient, normalized by the resolution
        array = elevation_ds.array.astype(np.float64)
        array[array==elevation_ds.nodata] = np.nan
        grad = np.gradient(array, elevation_ds.res.x, elevation_ds.res.y)
        
        # Find the length of the longest path uphill
        stack = np.dstack(grad)
        h = np.linalg.norm(stack, axis=2)
        
        # Compute the angle
        slope_rad = np.arctan(h)

        # Convert to degrees
        slope = slope_rad / np.pi * 180.
        slope[np.isnan(slope)] = elevation_ds.nodata
        
        slope_ds = GDALDataset(self.paths['slope'],
                               template=elevation_ds)
        slope_ds.array = slope
        return {'slope': slope_ds}


class SoilDepthOp(Operation):
    """ Compute soil depth from slope, elevation, and source area"""

    name = 'Soil Depth'
    op_id = 'soil-depth'
    output_types = ['soil-depth']

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
        soil_depth_ds = GDALDataset(self.paths['soil-depth'], template=elev_ds)
        soil_depth_ds.array = soil_depth_arr
        return {'soil-depth': soil_depth_ds}


class RasterToShapefileOp(Operation):
    """ Convert a raster to a shapefile """
    
    name = 'Raster to Shapefile'
    op_id = 'raster-to-shapefile'
    output_type = 'shapefile'
    filename_format = '{case_id}_{output_type}'

    def run(self, path, input_ds):
        # Convert the input raster to 0 and 1 mask
        # (A hack to make up for poor labeling by LabelGages)
        array = input_ds.array
        array[array==0] = 1
        array[array!=1] = 0
        input_ds.array = array

        # create the spatial reference, WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # Add raster boundary to shapefile
        input_band = input_ds.dataset.GetRasterBand(1)
        logging.debug(self.path.path)
        output_ds = BoundaryDataset(self.path, update=True).new()
        output_layer = output_ds.dataset.CreateLayer(
            'polygonized_raster', srs = srs)
        gdal.Polygonize(input_band, input_band,
                        output_layer, -1, [], callback=None)
        output_ds = None

        output_ds = BoundaryDataset(self.path)
        return output_ds

class LatLonToShapefileOp(Operation):
    """ Create a shapefile with a single point """
    
    name = 'Coordinate to Shapefile'
    op_id = 'coordinate-to-shapefile'
    output_type = 'shapefile'

    def run(self, path, coordinate, idstr='coordinate'):
        # Get spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(coordinate.epsg)

        # Create layer
        output_ds = BoundaryDataset(self.path, update=True).new()
        output_layer = output_ds.dataset.CreateLayer(
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
        output_ds.dataset.Destroy()
        output_ds = BoundaryDataset(self.path)
        return output_ds

    
class UpscaleFlowDirectionOp(Operation):

    name = 'Final Resolution Flow Direction'
    op_id = 'upscale-flow-direction'
    output_types = ['flow-direction']

    @property
    def resolution(self):
        return self.kwargs['template_ds'].resolution
    
    def run(self, flow_acc_ds, template_ds):
        flowgen_path = ScriptPath('flowgen', filename='flowgen').configure(
                                  self.run_config)
        flow_acc_path = Path(
                filename=flow_acc_ds.filepath.filename + '_nohead').configure(
                        self.run_config)
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
                         self.paths['flow-direction'].asc,
                         str(int(round(upscale_num.y))),
                         str(int(round(upscale_num.x))), '-v'])

        # Load ascii data into dataset with spatial reference
        flow_dir_ds = GDALDataset(self.paths['flow-direction'], 
                                  template=template_ds)
        flow_dir_array = np.loadtxt(self.paths['flow-direction'].asc)
        
        # Flip array back
        flow_dir_array = np.flipud(flow_dir_array)

        # Adjust output values
        flow_dir_array[abs(flow_dir_array - 4.5) > 3.5] = -9999
        flow_dir_array[np.isnan(flow_dir_array)] = -9999
        flow_dir_ds.nodata = -9999
        flow_dir_ds.array = flow_dir_array
        
        return {'flow-direction': flow_dir_ds}

    
class ConvertOp(Operation):
    
    def run(self, path, flow_dir_ds):
        out_ds = GDALDataset(self.path, template=flow_dir_ds)
        convert_array = np.vectorize(self.convert)

        input_array = np.ma.masked_where(flow_dir_ds.array==flow_dir_ds.nodata,
                                         flow_dir_ds.array)
        input_array.set_fill_value(flow_dir_ds.nodata)
        converted = convert_array(input_array)
        out_ds.array = converted.filled()
        return out_ds

    
class RVICToTauDEMOp(ConvertOp):
    
    name = 'Converted between VIC and TauDEM Flow Directions'
    op_id = 'rvic-to-taudem'
    output_type = 'tauDEM_flow_dir'

    def convert(self, array):
        return (3 - array) % 8 + 1

    
class BasinIDOp(Operation):
    '''
    Generates an RVIC-acceptable basin ID file from a mask

    This is a placeholder to actually computing the basin ID with TauDEM
    Gage Watershed or others - it will not work with multiple basins.
    '''
    name = 'Basin ID'
    op_id = 'basin-id'
    output_type = 'basin_id'

    def run(self, path, mask_ds):
        basin_id_ds = GDALDataset(self.path, template=mask_ds)
        
        # Compute basin ID array
        basin_id_ds.array[np.where(basin_id_ds.array==1)] = 1
        basin_id_ds.array[np.where(basin_id_ds.array!=1)] = basin_id_ds.nodata
        return basin_id_ds

class MeltNetCDF(Operation):
    '''
    Generates a CSV file with values for each coordinate in a raster
    '''
    name = 'Melted Raster'
    op_id = 'melt-nc'
    output_type = 'melted_raster'

    def run(self, path, raster_ds, variable):
        with open(raster_ds.filepath.nc) as raster_file:
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

        return True

class Melt(Operation):
    '''
    Generates a CSV file with values for each coordinate in a raster
    '''
    name = 'Melted Raster'
    op_id = 'melt'
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
        return True

class ShadingOp(Operation):
    """ Compute Slope and Aspect from a DEM """

    name = 'Shading'
    op_id = 'shading'
    output_type = 'shading'

    def run(self, path, elevation_ds):
        solar_path = ScriptPath('run_solar_programs_monthly.scr',
                                **self.run_config)

        # Convert to ASCII
        elev_path = TempPath(flow_acc_ds.filepath.filename + '_nohead',
                                 **self.run_config)
        elev_ds_long = elevation_ds.array.astype(np.float_)
        with open(elev_path.asc, 'w') as elev_file:
            elev_ds_long.tofile(elev_file)

        # Generate files
        lat = elevation_ds.center.lat
        lon = elevation_ds.center.lon
        cell_size = elevation_ds.res.x
        rows = elevation_ds.size.y
        cols = elevation_ds.size.x
        subprocess.call([solar_path.no_ext,
                         elev_path.asc,
                         self.run_config['basin_id'],
                         lat, lon, cell_size, rows, cols,
                         self.paths['solar']])
        return 