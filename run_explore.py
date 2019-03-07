import logging
import yaml
import csv
from osgeo import ogr, osr
import shutil, os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

from calibrate import CaseCollection
from utils import CoordProperty
from operation import LatLonToShapefileOp
from sequence import seqs
from path import TempPath

if __name__ == '__main__':
    # Load configuration file
    cfg = yaml.load(
            open('../landslide/i5/i5_gages_cfg.yaml',
                 'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.debug(cfg)

    cfg['case_id']  = 'common'
    collection = CaseCollection(cfg)
    cases = collection.cases
    case_main = cases[0]
    
    # Clear and create directory for result files
    temp_path = TempPath('', **cfg)
    if os.path.exists(temp_path.dirname):
        shutil.rmtree(temp_path.dirname)
    os.mkdir(temp_path.dirname)

    # Crop raster at perimeter if supplied
    if 'perimeter' in case_main.dir_structure.datasets:
        case_main.dir_structure.update_datasets(
                seqs.crop.run(case_main), seqs.crop.idstr)
    """
    # Prepare general files for delineation
    case_main.dir_structure.update_datasets(
            seqs.predelineate.run(case_main), seqs.predelineate.idstr)
    

    basemap = case_main.dir_structure.datasets['crop::low-res-dem'].plot(
        'Santa Barbara Fire', 'gages', show=False, cmap='Greys')
    basemap.drawcoastlines(linewidth=1.0, color='gray')
    """
        
    dsmin = case_main.dir_structure.datasets['crop::low-res-dem'].min
    dsmax = case_main.dir_structure.datasets['crop::low-res-dem'].max
    coords = {}
    with open(case_main.dir_structure.files['gages']) as gage_file:
        gage_reader = csv.DictReader(filter(lambda row: row[0]!='#',
                                            gage_file),
                                     delimiter='\t')
        coord = None
        dtypes = gage_reader.next()
        for row in gage_reader:
            try:
                lat = float(row['dec_lat_va'])
                lon = float(row['dec_long_va'])
            except ValueError:
                continue
            if (lat < dsmax.lat and lat > dsmin.lat
                    and lon < dsmax.lon and lon > dsmin.lon):
                coords[row['site_no']] = CoordProperty(
                    lat=float(row['dec_lat_va']),
                    lon=float(row['dec_long_va']))

    # Get GAGESII data to evaluate basin
    columns = ['STAID', 'CLASS', 'HYDRO_DISTURB_INDX', 'DRAIN_SQKM',
               'WR_REPORT_REMARKS', 'SCREENING_COMMENTS',
               'RUNAVE7100',
               'NDAMS_2009', 'DDENS_2009',
               'STOR_NID_2009', 'STOR_NOR_2009',
               'MAJ_NDAMS_2009', 'MAJ_DDENS_2009',
               'CANALS_PCT', 'CANALS_MAINSTEM_PCT',
               'FRESHW_WITHDRAWAL', 'MINING92_PCT', 'PCT_IRRIG_AG',
               'POWER_NUM_PTS', 'POWER_SUM_MW',
               'wy1995', 'wy1996', 'wy1997', 'wy1998', 'wy1999', 'wy2000',
               'wy2001', 'wy2002', 'wy2003', 'wy2004', 'wy2005', 'wy2006',
               'wy2007', 'wy2008', 'wy2009']
    class_df = pd.read_csv(case_main.dir_structure.files['basin_class'])
    class_df = class_df.set_index('STAID')
    id_df = pd.read_csv(case_main.dir_structure.files['basin_id'])
    id_df = id_df.set_index('STAID')
    hydro_df = pd.read_csv(case_main.dir_structure.files['hydro'])
    hydro_df = hydro_df.set_index('STAID')
    dam_df = pd.read_csv(case_main.dir_structure.files['hydromod_dams'])
    dam_df = dam_df.set_index('STAID')
    mod_df = pd.read_csv(case_main.dir_structure.files['hydromod_other'])
    mod_df = mod_df.set_index('STAID')
    fr_df = pd.read_csv(case_main.dir_structure.files['flow_record'])
    fr_df = fr_df.set_index('STAID')

    basin_ids = [int(id) for id in coords.keys()]
    basin_df = class_df.join(dam_df)
    basin_df = basin_df.join(id_df)
    basin_df = basin_df.join(hydro_df)
    basin_df = basin_df.join(mod_df)
    basin_df = basin_df.join(fr_df)
    basin_df = basin_df[basin_df.index.isin(basin_ids)]
    basin_df = basin_df.filter(items = columns)
    basin_df.to_csv(case_main.dir_structure.files['gagesii'])

    # Keep track of which colors have been used
    i = 0
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(coords))))
    shapes = {}

    gfilt_path = case_main.dir_structure.paths['nearby-gages']
    print gfilt_path.path
    if not gfilt_path.isfile:
        # Load GAGESII
        gii_uri = '/Volumes/LabShare/GAGESII_reorganize/CONUS/bas_all_us.shp'
        logging.info('Loading Gages II file')
        
        gii_ds = ogr.Open(gii_uri)
        gii_lyr = gii_ds.GetLayer()
        
        logging.info('Filtering Gages II file')
        gii_lyr.SetAttributeFilter(
            'GAGE_ID IN {}'.format(tuple(coords.keys())))
        gii_lyrdef = gii_lyr.GetLayerDefn()
        gii_srs = gii_lyr.GetSpatialRef()

        # Create the output dataset
        esri_driver = ogr.GetDriverByName("ESRI Shapefile")
        gfilt_ds = esri_driver.CreateDataSource(gfilt_path.shp)

        # create the layer in spatial reference WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        gfilt_lyr = gfilt_ds.CreateLayer('GAGESII', srs,
                                         ogr.wkbMultiPolygon)

        # Add input Layer Fields to the output Layer
        # if it is the one we want
        for i in range(0, gii_lyrdef.GetFieldCount()):
            field_def = gii_lyrdef.GetFieldDefn(i)
            gfilt_lyr.CreateField(field_def)
    
        # Add features to the output Layer from the filtered layer
        gfilt_lyrdef = gfilt_lyr.GetLayerDefn()
        transform =  osr.CoordinateTransformation(gii_srs, srs)
        logging.info('Writing filtered Gages II file')
        for gii_feature in gii_lyr:
         # Create output Feature
            gfilt_feature = ogr.Feature(gfilt_lyrdef)

            # Add field values from input Layer
            for i in range(0, gfilt_lyrdef.GetFieldCount()):
                field_def = gfilt_lyrdef.GetFieldDefn(i)
                field_name = field_def.GetName()
                field_val = gii_feature.GetField(i)
                gfilt_feature.SetField(
                        gfilt_lyrdef.GetFieldDefn(i).GetNameRef(),
                        gii_feature.GetField(i))

            # Set geometry
            geom = gii_feature.GetGeometryRef()
            geom.Transform(transform)
            gfilt_feature.SetGeometry(geom.Clone())
        
            # Add new feature to output Layer
            gfilt_lyr.CreateFeature(gfilt_feature)

        # Close DataSources
        gii_ds.Destroy()
        gfilt_ds.Destroy()
    """
    # Read in Gages II boundaries
    #basemap.readshapefile(gfilt_path.no_ext, 'gagesii', drawbounds=False)
    case_main.dir_structure.update_datasets({'merge::low-res-dem':
                    case_main.dir_structure.datasets['crop::low-res-dem']})

    for case_id, coord in coords.iteritems():
        # Generate case configuration
        cfg['case_id'] = case_id
        collection = CaseCollection(cfg)
        cases = collection.cases
        case = None
        case = cases[0]
        case.dir_structure.update_datasets(case_main.dir_structure.datasets)

        outlet_path = case.dir_structure.paths['outlet']
        case.dir_structure.datasets['outlet'] = LatLonToShapefileOp(
                cfg, coordinate=coord, idstr=str(cfg['case_id']),
                path=outlet_path
        ).saveas('shp', working_ext='shp')
        
        # Delineate
        seqs.delineate.run(case)

        # Plot watershed
    """
    """
        color = next(colors)
        basemap.readshapefile(
                case.dir_structure.paths['delineate::boundary'].no_ext,
                case_id, drawbounds=False)
        shape =  getattr(basemap, case_id)
        as_array = np.array([[p[0] for p in shape[0]],
                             [p[1] for p in shape[0]]]).T
        poly = Polygon(as_array, closed=True,
                       facecolor=color, alpha=0.5, zorder=2)
        plt.gca().add_artist(poly)
    
        # Save polygon to place in legend
        shapes[case_id] = poly

        # Plot gage location
        x, y = basemap([coord.lon], [coord.lat])
        plt.scatter(x, y, 10, marker='D', color=color, zorder=3,
                        edgecolors='k')
        
        i += 1
    """
    """
        # Clean up
        tmp_fns = glob.glob('{}*'.format(
                os.path.join(temp_path.dirname, case_id)))
        for filename in tmp_fns:
            os.remove(filename)

    """
    """
    for info, shape in zip(basemap.gagesii_info, basemap.gagesii):
            lon, lat = zip(*shape)
            basemap.plot(lon, lat, marker=None, color='k', linewidth=1.5,
                         zorder=3)
    
    # Matplotlib is failing at the moment and this runs in R anywayde
    basemap.readshapefile(case.dir_structure.paths['perimeter'].no_ext,
                          'thomas_fire', drawbounds=False)
    fire_shape =  getattr(basemap, 'thomas_fire')
    fire_as_array = np.array([[p[0] for p in fire_shape[0]],
                             [p[1] for p in fire_shape[0]]]).T
    poly = Polygon(fire_as_array, closed=True, edgecolor='k', linewidth=0.2,
                   facecolor='white', alpha=0.5, zorder=4)
    plt.gca().add_artist(poly)


            
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([shapes[case_id] for case_id in coords.keys()])
    labels.extend(coords.keys())
    plt.legend(handles=handles, labels=labels, framealpha=1.)
    
    plt.show()
    """
