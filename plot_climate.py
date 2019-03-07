from netCDF4 import Dataset, num2date
import glob
import pandas as pd
import re
import datetime
import matplotlib.pyplot as plt
import numpy as np

def load_spatial_data(pattern, case_group, var_name='streamflow'):
    files = glob.glob(pattern)
    dfs = []
    for filename in files:
        delta = re.search(
                r'({grp}_.*)/rvic'.format(grp=case_group), filename).group(1)
        data = Dataset(filename)
        var = data.variables[var_name]
        var = np.sum(var, axis=(1, 2))
        times = data.variables['time']
        jd = num2date(times[:], times.units)
        dataframe = pd.DataFrame({delta: var, 'date': jd})
        dataframe['Month'] = pd.to_datetime(dataframe['date']).map(
            lambda dt: dt.replace(day=15))
        dataframe.groupby('Month')[delta].mean().reset_index()
        print(dataframe)
        dfs.append(dataframe)
    return reduce(lambda x,y: pd.merge(x, y, on='Month'), dfs)

def load_data(pattern, case_group, var_name='streamflow'):
    files = glob.glob(pattern)
    dfs = []
    for filename in files:
        delta = re.search(
                r'({grp}_.*)/rvic'.format(grp=case_group), filename).group(1)
        data = Dataset(filename)
        var = data.variables[var_name][:,0]
        times = data.variables['time']
        jd = num2date(times[:], times.units)
        dataframe = pd.DataFrame({var_name: var, 'date': jd})
        print(dataframe)
        dfs.append(dataframe)
    return reduce(lambda x,y: pd.merge(x, y), dfs)

def plot_sediment_cumsums():
    path = 'cases/or/calibrate_dab9cf80ae1541a7b8d3eded1882af9b/rvic/runoff.baseflow.convolution/hist/calibrate_dab9cf80ae1541a7b8d3eded1882af9b.rvic.h0a.1991-08-30.nc'
    data = Dataset(path)
    streamflow = data.variables['streamflow'][:,0]
    times = data.variables['time']
    jd = num2date(times[:], times.units)
    dataframe = pd.DataFrame({'reference': streamflow, 'Date': jd})
    dataframe.set_index('Date')
    dataframe.index = pd.to_datetime(dataframe.index)
    
    path = 'cases/or/calibrate_dab9cf80ae1541a7b8d3eded1882af9b/rvic/dhsvm.convolution/hist/calibrate_dab9cf80ae1541a7b8d3eded1882af9b.rvic.h0a.1991-08-30.nc'
    data = Dataset(path)
    streamflow = data.variables['streamflow'][:,0]
    times = data.variables['time']
    jd = num2date(times[:], times.units)
    dataframe2 = pd.DataFrame({'reference': streamflow, 'Date': jd})
    dataframe2.set_index('Date')
    dataframe2.index = pd.to_datetime(dataframe2.index)
    
    # DHSVM Precip
    dhsvm_conv = load_data(r'cases/or/precipitation*/rvic/dhsvm.convolution/hist/precipitation*.nc', 'precipitation')

    dhsvm_cs = dhsvm_conv.drop('Date', axis=1)
    dhsvm_cs = dhsvm_cs.cumsum(axis=0)
    dhsvm_cs.index = dhsvm_conv['Date']
    ax2 = dhsvm_cs.plot(fontsize=18)
    ax2.set_title('DHSVM - Modified Precipitation', fontsize=24)
    ax2.set_ylabel('Sediment (kg/m2/s)', fontsize=20)
    ax2.set_xlabel('Date', fontsize=20)
    
    dataframe2.index = dataframe2['Date']
    control2 = dataframe2['1960-10-01':'1970-09-30']
    control2['reference'] = control2['reference'].cumsum(axis=0)
    control2.plot(ax=ax2, color='black')

    # MRC Precip
    streamflow_conv = load_data('cases/or/precipitation*/rvic/baseflow.runoff.convolution/hist/precipitation*.nc', 'precipitation')
    sediment_mrc = 18.23422783*streamflow_conv.drop('Date', axis=1)**0.41839178
    sediment_mrc = sediment_mrc.cumsum(axis=0)
    sediment_mrc.index = streamflow_conv['Date']
    ax1 = sediment_mrc.plot(fontsize=18)
    ax1.set_title('MRC - Modified Precipitation', fontsize=24)
    ax1.set_ylabel('Sediment (ac-ft)', fontsize=20)
    ax1.set_xlabel('Date', fontsize=20)
    
    dataframe.index = dataframe['Date']
    control = dataframe['1960-10-01':'1970-09-30']
    control['reference'] = 18.23422783 * control['reference'] ** 0.41839178
    control['reference'] = control['reference'].cumsum(axis=0)
    control.plot(ax=ax1, color='black')

    # DHSVM Tmax
    t_dhsvm_conv = load_data(r'cases/or/tmax*/rvic/dhsvm.convolution/hist/tmax*.nc', 'tmax')

    t_dhsvm_cs = t_dhsvm_conv.drop('Date', axis=1)
    t_dhsvm_cs = t_dhsvm_cs.cumsum(axis=0)
    t_dhsvm_cs.index = t_dhsvm_conv['Date']
    ax3 = t_dhsvm_cs.plot(fontsize=18)
    ax3.set_title('DHSVM - Modified Temperature', fontsize=24)
    ax3.set_ylabel('Sediment (kg/m2/s)', fontsize=20)
    ax3.set_xlabel('Date', fontsize=20)

    control2.plot(ax=ax3, color='black')

    # MRC Tmax
    t_streamflow_conv = load_data('cases/or/tmax*/rvic/baseflow.runoff.convolution/hist/tmax*.nc', 'tmax')
    t_sediment_mrc = 18.23422783*t_streamflow_conv.drop('Date', axis=1)**0.41839178
    t_sediment_mrc = t_sediment_mrc.cumsum(axis=0)
    t_sediment_mrc.index = t_streamflow_conv['Date']
    ax4 = t_sediment_mrc.plot(fontsize=18)
    ax4.set_title('MRC - Modified Temperature', fontsize=24)
    ax4.set_ylabel('Sediment (ac-ft)', fontsize=20)
    ax4.set_xlabel('Date', fontsize=20)
    control.plot(ax=ax4, color='black')
    plt.show()

def plot_latent():
    precip_data = load_spatial_data(
            'cases/or/precipitation_*_LH/rvic/input/forcing/*.nc',
            'precipitation', var_name='latent')
    tmax_data = load_spatial_data(
            'cases/or/tmax_*_LH/rvic/input/forcing/*.nc',
            'tmax', var_name='latent')
    ax = precip_data.plot()
    ax.set_title('Latent Heat Fluxes - four climate scenarios', fontsize=20)
    ax.set_ylabel('Latent Heat ()', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    tmax_data.plot(ax=ax)
    plt.show()
    
if __name__ == '__main__':
    #plot_sediment_cumsums()
    plot_latent()
    
