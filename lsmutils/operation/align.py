import copy
import gdal
import logging
import ogr
import osr
import pandas as pd
import re
import subprocess

from lsmutils.dataset import BoundaryDataset
from lsmutils.operation import Operation, OutputType
from lsmutils.utils import CoordProperty, BBox

class ConvertFileType(Operation):

    title = 'Convert Filetype'
    name = 'convert-filetype'
    output_types = [OutputType('converted', '')]

    def run(self, input_ds, filetype):
        self.locs['converted'].default_ext = filetype
        gdal.Translate(self.locs['converted'].path,
                       input_ds.dataset,
                       format=input_ds.filetypes[filetype])


class MosaicOp(Operation):

    title = 'Mosaic Rasters with gdal_merge'
    name = 'mosaic'
    output_types = [OutputType('merged', 'tif')]

    def run(self, input_ds):
        if not hasattr(input_ds, '__iter__'):
            input_ds = [input_ds]
        merge_args = [
            'gdal_merge.py',
            '-o', self.locs['merged'].path,
            *[ds.loc.path for ds in input_ds]
        ]
        logging.info('Calling process %s', ' '.join(merge_args))
        merge_process = subprocess.Popen(
            merge_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        merge_output, _ = merge_process.communicate()
        logging.info(merge_output)

class MergeOp(Operation):

    title = 'Merge rasters into a virtual raster'
    name = 'merge'
    output_types = [OutputType('merged', 'vrt')]

    def run(self, input_ds):
        if not hasattr(input_ds, '__iter__'):
            input_ds = [input_ds]
        vrt_args = [
            'gdalbuildvrt',
            self.locs['merged'].path,
            *[ds.loc.path for ds in input_ds]
        ]
        logging.info('Calling process %s', ' '.join(vrt_args))
        vrt_process = subprocess.Popen(
            vrt_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        vrt_output, _ = vrt_process.communicate()
        logging.info(vrt_output)

class MatchRasterOp(Operation):

    title = 'Warp dataset to match a template raster'
    name = 'match-raster'
    output_types = [OutputType('matched', 'gtif')]

    def run(self, input_ds, template_ds=None, algorithm='bilinear'):
        agg_warp_options = gdal.WarpOptions(
            outputBounds = template_ds.rev_warp_output_bounds,
            width = template_ds.size.x,
            height = template_ds.size.y,
            resampleAlg=algorithm,
        )
        gdal.Warp(
            self.locs['matched'].path,
            input_ds.dataset,
            options=agg_warp_options)


class GridAlignOp(Operation):

    title = 'Align Dataset'
    name = 'grid-align'
    output_types = [OutputType('aligned', 'gtif')]

    def run(self, input_ds,
            template_ds=None, bbox=None,
            resolution=CoordProperty(x=1/240., y=1/240.), grid_res=None,
            padding=CoordProperty(x=0, y=0),
            algorithm='bilinear'):
        if not grid_res:
            grid_res = resolution
        if template_ds:
            resolution = template_ds.resolution
            bbox = copy.copy(template_ds.bbox)
        if bbox:
            grid_box = [bbox.llc.x - padding.x, bbox.llc.y - padding.y,
                        bbox.urc.x + padding.x, bbox.urc.y + padding.y]
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
        gdal.Warp(
            self.locs['aligned'].path,
            input_ds.dataset,
            options=agg_warp_options)


class CropOp(Operation):

    title = 'Crop Raster Dataset'
    name = 'crop'
    output_types = [OutputType('cropped', 'gtif')]

    def run(self, input_ds,
            template_ds=None, bbox=None,
            padding=CoordProperty(x=0, y=0),
            algorithm='bilinear'):
        if template_ds:
            bbox = copy.copy(template_ds.bbox)
            # SRS needs to match for output bounds and input dataset
            if not template_ds.srs == input_ds.srs:
                transform = osr.CoordinateTransformation(
                    template_ds.srs, input_ds.srs)
                bbox = BBox(
                    llc = CoordProperty(
                        *transform.TransformPoint(
                            bbox.llc.x, bbox.llc.y)[:-1]),
                    urc = CoordProperty(
                        *transform.TransformPoint(
                            bbox.urc.x, bbox.urc.y)[:-1])
                )


        grid_box = [
            bbox.llc.x - padding.x,
            bbox.llc.y - padding.y,
            bbox.urc.x + padding.x,
            bbox.urc.y + padding.y
        ]

        agg_warp_options = gdal.WarpOptions(
            outputBounds = grid_box,
            resampleAlg=algorithm,
        )
        gdal.Warp(
            self.locs['cropped'].path,
            input_ds.dataset,
            options=agg_warp_options)


class ClipOp(Operation):

    title = 'Raster clipped to boundary'
    name = 'clip'
    output_types = [OutputType('clipped_raster', 'gtif')]

    def run(self, input_ds, boundary_ds, algorithm='bilinear'):
        clip_warp_options = gdal.WarpOptions(
            format='GTiff',
            cutlineDSName=boundary_ds.loc.shp,
            cutlineBlend=.5,
            dstNodata=input_ds.nodata,
            resampleAlg=algorithm
        )
        gdal.Warp(path, input_ds.dataset, options=clip_warp_options)


class ReprojectRasterOp(Operation):

    title = 'Reproject raster'
    name = 'reproject-raster'
    output_types = [OutputType('reprojected', 'tif')]

    def run(self, input_ds, template_ds=None, srs=None, algorithm='bilinear'):
        if template_ds:
            srs = template_ds.srs
        agg_warp_options = gdal.WarpOptions(
            dstSRS = srs,
            resampleAlg=algorithm)
        gdal.Warp(
            self.locs['reprojected'].path,
            input_ds.dataset,
            options=agg_warp_options)

class ReprojectVectorOp(Operation):

    title = 'Reproject Vector Dataset'
    name = 'reproject-vector'
    output_types = [OutputType('reprojected', 'shp')]

    def run(self, input_ds, srs):
        reproj_ds = BoundaryDataset(
                self.locs['reprojected'], update=True).new()

        out_srs = osr.SpatialReference()
        if not srs.startswith('EPSG:'):
            raise ValueError('SRS definition must start with "EPSG:"')
        out_srs.ImportFromEPSG(int(srs[5:]))

        for layer in input_ds.layers:
            # SRS transform
            in_srs = layer.GetSpatialRef()
            transform = osr.CoordinateTransformation(in_srs, out_srs)

            # create the output layer
            reproj_layer = reproj_ds.dataset.CreateLayer(
                layer.GetName(), srs=out_srs, geom_type=layer.GetGeomType())

            # add fields
            layer_defn = layer.GetLayerDefn()
            for i in range(0, layer_defn.GetFieldCount()):
                field_defn = layer_defn.GetFieldDefn(i)
                reproj_layer.CreateField(field_defn)

            # loop through the input features
            reproj_layer_defn = reproj_layer.GetLayerDefn()
            feature = layer.GetNextFeature()
            while feature:
                # reproject the input geometry
                geom = feature.GetGeometryRef()
                geom.Transform(transform)

                # create a new feature with same geometry and attributes
                reproj_feature = ogr.Feature(reproj_layer_defn)
                reproj_feature.SetGeometry(geom)
                for j in range(0, reproj_layer_defn.GetFieldCount()):
                    reproj_feature.SetField(
                        reproj_layer_defn.GetFieldDefn(j).GetNameRef(),
                        feature.GetField(j))

                # add the feature to the shapefile
                reproj_layer.CreateFeature(reproj_feature)

                # dereference the features and get the next input feature
                reproj_feature = None
                feature = layer.GetNextFeature()

        # Save and close the shapefiles
        input_ds.close()
        reproj_ds.close()
