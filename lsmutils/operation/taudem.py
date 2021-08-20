import copy
import csv
import glob
import numpy as np
import os
from osgeo import ogr, osr
import pandas as pd
import re
import shutil
import string
import subprocess

from lsmutils.operation import Operation, OutputType
from lsmutils.dataset import GDALDataset, BoundaryDataset


class RemoveSinksOp(Operation):

    title = 'Sinks Removed'
    name = 'remove-sinks'
    output_types = [OutputType('no-sinks', 'tif')]

    def run(self, input_ds):
        # Remove Pits / Fill sinks
        subprocess.call(['pitremove',
                         '-z', input_ds.loc.path,
                         '-fel', self.locs['no-sinks'].no_ext])


class FlowDirectionOp(Operation):

    title = 'Flow Direction'
    name = 'flow-direction'
    output_types = [
        OutputType('flow-direction', 'tif'),
        OutputType('slope', 'tif')
    ]

    def run(self, dem_ds):
        subprocess.call([
            'd8flowdir',
            '-fel', dem_ds.loc.path,
            '-p', self.locs['flow-direction'].path,
            '-sd8', self.locs['slope'].path
        ])


class SourceAreaOp(Operation):

    title = 'Source Area/Flow Accumulation'
    name = 'source-area'
    output_types = [OutputType('source-area', 'tif')]

    def run(self, flow_dir_ds):
        subprocess.call([
            'aread8',
            '-p', flow_dir_ds.loc.path,
            '-ad8', self.locs['source-area'].no_ext,
            '-nc'
        ])
        source_area_ds = GDALDataset(self.locs['source-area'])
        source_area_ds.nodata = 0


class StreamDefinitionByThresholdOp(Operation):
    """ Run the TauDEM Stream Definition By Threshold Command """

    title = 'Stream Definition By Threshold'
    name = 'stream-definition-threshold'
    output_types = [OutputType('stream-raster', 'tif')]

    def run(self, source_area_ds):
        threshold = np.percentile(source_area_ds.array, 98)
        subprocess.call(['threshold',
                         '-ssa', source_area_ds.loc.path,
                         '-thresh', '{:.1f}'.format(threshold),
                         '-src', self.locs['stream-raster'].no_ext])


class MoveOutletsToStreamOp(Operation):
    """ Run the TauDEM Move Outlets to Streams Command """

    title = 'Move Outlets to Streams'
    name = 'snap-outlet'
    output_types = [
        OutputType('outlet-on-stream-nosrs', 'shp'),
        OutputType('outlet-on-stream', 'shp')
    ]

    def run(self, flow_dir_ds, stream_ds, outlet_ds):
        subprocess.call(['moveoutletstostrm',
                         '-p', flow_dir_ds.loc.path,
                         '-src', stream_ds.loc.path,
                         '-o', outlet_ds.loc.path,
                         '-om', self.locs['outlet-on-stream-nosrs'].path])

        # Copy spatial reference from original outlet
        in_ds = BoundaryDataset(self.locs['outlet-on-stream-nosrs'])
        out_ds = BoundaryDataset(self.locs['outlet-on-stream'],
                                 update=True).new()

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


class LabelGagesOp(Operation):
    """ Add a sequential id field to each outlet in a shapefile """

    title = 'Labeled Gages'
    name = 'label-outlet'
    output_types = [OutputType('labelled-outlet', 'shp')]

    def run(self, outlet_ds):
        ## Fix this - modify dataset at the new location, not the old
        outlet_ds.chmod(True)
        outlet_path = outlet_ds.loc

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
            shutil.copyfile(
                a_file,
                self.locs['labelled-outlet'].ext(
                    os.path.splitext(a_file)[1][1:]))


class GageWatershedOp(Operation):
    """
    Run the TauDEM Gage Watershed Command

    Raster labeling each point by which gage it drains to directly
    """

    title = 'Gage Watershed'
    name = 'gage-watershed'
    output_types = [OutputType('gage-watershed', 'tif')]

    def run(self, flow_dir_ds, outlet_ds):
        subprocess.call(['gagewatershed',
                         '-p', flow_dir_ds.loc.path,
                         '-o', outlet_ds.loc.path,
                         '-gw', self.locs['gage-watershed'].path])


class PeukerDouglasStreamDefinitionOp(Operation):
    """ Run the TauDEM Peuker Douglas Stream Definition Command """

    title = 'Peuker Douglas Stream Definition'
    name = 'stream-def-pd'
    output_types = [
        OutputType('ssa', 'tif'),
        OutputType('drop-analysis', 'txt'),
        OutputType('stream-definition', 'tif')
    ]

    def run(self, no_sinks_ds, flow_dir_ds, source_area_ds, outlet_ds):
        # The threshold range should be selected based on the raster size
        # Something like 10th to 99th percentile of flow accumulation?

        ## This is a three-step process - first compute the D8 source area
        subprocess.call(['aread8',
                         '-p', flow_dir_ds.loc.path,
                         '-o', outlet_ds.loc.path,
                         '-ad8', self.locs['ssa'].path])

        ## Next perform the drop analysis
        # This selects a sensible flow accumulation threshold value
        thresh_min = np.percentile(source_area_ds.array, 95)
        thresh_max = np.max(source_area_ds.array)

        subprocess.call(['dropanalysis',
                         '-p', flow_dir_ds.loc.path,
                         '-fel', no_sinks_ds.loc.path,
                         '-ad8', source_area_ds.loc.path,
                         '-o', outlet_ds.loc.path,
                         '-ssa', self.locs['ssa'].path,
                         '-drp', self.locs['drop-analysis'].path,
                         '-par', str(thresh_min), str(thresh_max), '40', '0'])

        ## Finally define the stream
        # Extract the threshold from the first row with drop statistic t < 2
        with open(self.locs['drop-analysis'].path, 'r') as drop_file:
            # Get optimum threshold value from last line
            for line in drop_file:
                pass
            last = line
            thresh_re = re.compile(r'([\.\d]*)$')
            threshold = thresh_re.search(last).group(1)

        subprocess.call(['threshold',
                         '-ssa', self.locs['ssa'].path,
                         '-thresh', threshold,
                         '-src', self.locs['stream-definition'].path])


class DinfFlowDirOp(Operation):
    """ Compute Slope and Aspect from a DEM """

    title = 'D-infinity Flow Direction'
    name = 'dinf-flow-direction'
    output_types = [
        OutputType('slope', 'tif'),
        OutputType('aspect', 'tif')
    ]

    def run(self, elevation_ds):
        subprocess.call(['dinfflowdir',
                         '-fel', elevation_ds.loc.path,
                         '-slp', self.locs['slope'].path,
                         '-ang', self.locs['aspect'].path])

class StreamNetworkOp(Operation):
    """ Run the TauDEM Stream Reach and Watershed Command """

    title = 'Stream Network'
    name = 'stream-network'
    output_types = [
        OutputType('order', 'tif'),
        OutputType('tree', 'tsv'),
        OutputType('coord', 'tsv'),
        OutputType('network', 'shp'),
        OutputType('watershed', 'tif')
    ]

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

    def run(self, no_sinks_ds, flow_dir_ds, source_area_ds,
            pd_stream_def_ds, outlet_ds):

        subprocess.call([
            'streamnet',
            '-p', flow_dir_ds.loc.path,
            '-fel', no_sinks_ds.loc.path,
            '-ad8', source_area_ds.loc.path,
            '-src', pd_stream_def_ds.loc.path,
            '-o', outlet_ds.loc.path,
            '-ord', self.locs['order'].path,
            '-tree', self.locs['tree'].path,
            '-coord', self.locs['coord'].path,
            '-net', self.locs['network'].path,
            '-w', self.locs['watershed'].path])
        self.locs['tree'].csvargs.update({
            'delimiter': '\t', 'header': None, 'names': self.tree_colnames})
        self.locs['coord'].csvargs.update({
            'delimiter': '\t', 'header': None, 'names': self.coord_colnames})
