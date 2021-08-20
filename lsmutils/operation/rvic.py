import numpy as np
from osgeo import gdal, ogr, osr
import subprocess 

from lsmutils.loc import *
from lsmutils.operation import Operation, OutputType

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
