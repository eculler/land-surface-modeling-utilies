from scipy.optimize import minimize
from netCDF4 import Dataset, num2date
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import datetime


if __name__ == '__main__':
    path = 'cases/or/calibrate_dab9cf80ae1541a7b8d3eded1882af9b/rvic/runoff.baseflow.convolution/hist/calibrate_dab9cf80ae1541a7b8d3eded1882af9b.rvic.h0a.1991-08-30.nc'
    data = Dataset(path)
    conversion_factor =  8075090266 / 0.0283168 / 1000 / 232
    streamflow = data.variables['streamflow'][:,0]
    streamflow = streamflow #* conversion_factor
    print streamflow
    print sum(streamflow)
    def sediment(x):
        return abs(np.sum(x[0] * streamflow ** x[1]) - 4586)
    res = minimize(sediment, [10, 5], method='nelder-mead')

    sediment = res['x'][0] * streamflow ** res['x'][1]
    times = data.variables['time']
    jd = num2date(times[:], times.units)
    jd = [val.date() + datetime.timedelta(days=60) for val in jd]
    dataframe = DataFrame({'sediment': sediment, 'time': jd})
    dataframe.set_index('time')
    dataframe['sediment'] = dataframe['sediment'].cumsum(axis=0)
    
    ax = dataframe.plot(legend=False, fontsize=20)
    ax.axhline(4586)
    ax.set_xlabel('Date', fontsize=24)
    ax.set_ylabel('Sediment (ac-ft)', fontsize=24)
    plt.show()

    print res
