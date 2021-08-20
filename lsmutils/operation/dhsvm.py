import copy
import csv
import datetime
import geopandas as gpd
import logging
import netCDF4 as nc4
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units
import numpy as np
from osgeo import osr
import pandas as pd
import subprocess

from lsmutils.loc import *
from lsmutils.utils import CoordProperty, BBox
from lsmutils.operation import (
    Operation, OutputType,
    WriteOneVarNetCDFOp,
    WriteUniformNetCDFOp)

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
    out_columns = [
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
            nldas_ds, elevation_ds, projected_epsg,
            precip_ds = None, precip_var = '', precip_dt = '1H',
            precip_adj=1):
        station_ds = []

        time = nldas_ds.dataset.variables['time']
        start_i = nc4.date2index(start, time) - time_zone
        end_i = nc4.date2index(end, time) - time_zone
        datetimes = pd.date_range(start, end, freq=dt)

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
            'date_format': '%m/%d/%Y-%H:%M',
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
                columns = self.out_columns
            )

            ds.csvargs = csvargs
            df = ds.dataset
            for name, id in self.vars.items():
                slice = []
                for dim in nldas_ds.dataset[id].dimensions:
                    if dim == 'time':
                        # Include end datetime
                        slice.append('{start_i}:{end_i}'.format(
                            start_i=start_i, end_i=end_i+1))
                    elif dim == nldas_ds.xdim:
                        slice.append(str(meta['xi']))
                    elif dim == nldas_ds.ydim:
                        slice.append(str(meta['yi']))
                expr = "nldas_ds.dataset['{id}'][{slice}]".format(
                    id=id, slice=','.join(slice))
                logging.debug(expr)
                df[name] = eval(expr)

            df['wind_speed'] = np.sqrt(
                df['wind_speed_u']**2 + df['wind_speed_v']**2 )
            df['air_temp'] = df['air_temp'] - 273.15
            df['relative_humidity'] = (
                df.apply(
                    lambda row:(
                        100 * relative_humidity_from_specific_humidity(
                            units.Quantity(row['pressure'], "pascal"),
                            units.Quantity(row['air_temp'], "degC"),
                            units.Quantity(row['specific_humidity'],
                                           "dimensionless")
                            ).magnitude),
                    axis=1 ))
            df.loc[df.relative_humidity >= 100, 'relative_humidity'] = 100

            # mm/hour to m/hour, adjusted for calibration
            if precip_ds is None:
                df['precipitation'] = df['precipitation'] / 1000. * precip_adj
            else:
                # Resample the original meteorology to match precipitation
                df_p = df.resample(precip_dt).ffill()
                logging.debug(df)

                # Select the correct precipitation range and location
                # Convert start and end to indices
                slice = []
                p_time = precip_ds.dataset.variables['time']
                p_start_i = nc4.date2index(start, p_time) - time_zone
                p_end_i = nc4.date2index(end, p_time) - time_zone
                # Construct slice
                for dim in precip_ds.dataset[precip_var].dimensions:
                    if dim == 'time':
                        slice.append('{start_i}:{end_i}'.format(
                            start_i=p_start_i, end_i=p_end_i+1))
                    elif dim == precip_ds.xdim:
                        slice.append(str(meta['xi']))
                    elif dim == precip_ds.ydim:
                        slice.append(str(meta['yi']))
                expr = "precip_ds.dataset['{id}'][{slice}]".format(
                    id=precip_var, slice=','.join(slice))
                logging.debug(expr)
                df_p['precipitation'] = eval(expr)
                df_p['precipitation'] = df_p['precipitation'] / 1000. * precip_adj
                ds._dataset = df_p
                logging.debug(ds.dataset)

            ds.save()

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
