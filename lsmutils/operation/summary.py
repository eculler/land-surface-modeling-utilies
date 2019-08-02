import gdal
import logging
import pandas as pd

from lsmutils.operation import Operation, OutputType

class CropDataFrameOp(Operation):

    title = 'Merged Raster'
    name = 'merge'
    output_types = [OutputType('merged', 'tif')]

    def run(self, input_ds):
        out_file = 'processed/glc_swe_alldates.csv'
        if not os.path.exists(os.path.dirname(out_file)):
            raise FileNotFoundError('Directory does not exist: {}'.format(out_file))
        slide = pd.read_csv('../data/SLIDE_NASA_GLC/GLC20180821.csv',
                            index_col='OBJECTID')
        origen_x = -125
        origen_y = 32
        res = 0.05

         # Filter Landslides to study area and duration
        slide = slide[slide.latitude > 32]
        slide = slide[slide.latitude < 43]
        slide = slide[slide.longitude > -125]
        slide = slide[slide.longitude < -114]

        slide['event_date'] = pd.to_datetime(
                slide['event_date'], format='%Y/%m/%d %H:%M')
        slide['event_date'] = slide['event_date'].dt.normalize()
        slide = slide[slide.event_date >= '2004-01-01']
        slide = slide[slide.event_date < '2016-01-01']

        # Open swe files
        files = glob.glob('data/daymet/daymet_v3_swe_*_na.nc4')
        files.sort()
        arrays = [xr.open_dataset(file) for file in files]
        coords = arrays[0].isel(time=0).drop(['time'])

        # Filter coordinates
        coords = coords.where(
            (coords.lat < 43) & (coords.lat > 32) &
            (coords.lon > -125) & (coords.lon < -114), drop=True)

        # Build KD-Tree
        locs = list(zip(coords.lon.values.flatten(), coords.lat.values.flatten()))
        kdt = cKDTree(locs)
        xa, ya = np.meshgrid(coords.x.values, coords.y.values)
        xv = xa.flatten()
        yv = ya.flatten()

        # Add swe to dataframe
        event_dfs = []
        count = 0
        for index, row in slide.iterrows():
            print(count)
            count += 1
            # Pull out swe values for location
            loc_i = kdt.query((row.longitude, row.latitude))[1]
            event_swe = pd.concat([
                arr.sel(x=xv[loc_i], y=yv[loc_i])[['swe']].to_dataframe()
                for arr in arrays])

            # Add location identifier to the index
            event_swe['OBJECTID'] = index
            event_swe = event_swe.set_index(['OBJECTID'], append=True)
            event_dfs.append(event_swe)

        swe_df = pd.concat(event_dfs)

        swe_df.to_csv(out_file)
        print(swe_df.sort_index().head())
