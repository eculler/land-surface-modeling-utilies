import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import stats
from math import ceil, sqrt
import numpy as np
from netCDF4 import Dataset, num2date
import os, shutil
import datetime
import logging
import pandas as pds
from pandas import DataFrame, Series, read_csv, to_datetime, date_range
from pandas.plotting import parallel_coordinates
from sklearn import preprocessing

def load_rvic_data(cases_obj=None, case_id=None,
                   case_path=None):
    if case_id and cases_obj:
        # Get the convolution file path for this case id
        case = [case for case in cases if case.cfg['case_id']==case_id][0]
        ds = case.dir_structure
        hist_dir = os.path.join(ds.files['rvic-convolution-dir'], 'hist')
        hist_path = [os.path.abspath(os.path.join(hist_dir, x))
                     for x in os.listdir(hist_dir)][0]
    elif case_path:
        hist_path = os.path.join(case_path, os.listdir(case_path)[0])
    else:
        raise InputError('Requires BOTH case object and case id or BOTH' +
                          'case path and convolution type')

    # Read netCDF file
    dataset = Dataset(hist_path)
    streamflow = dataset.variables['streamflow'][:,0]
    times = dataset.variables['time']
    jd = num2date(times[:], times.units)
    jd = [val.date() + datetime.timedelta(days=60) for val in jd]
    dataframe = DataFrame({'streamflow': streamflow, 'time': jd})
    dataframe.set_index('time')
    # Convert to ac-ft
    conversion_factor = 86400 * 8075090266 / 232 * 0.000810714  #/ 0.0283168 / 1000 / 232
    dataframe['streamflow'] = dataframe['streamflow'] * conversion_factor
    return dataframe

def load_obs_data(obs_path):
    names =  ('agency', 'site', 'date', 'streamflow', 'code')
    dataframe = read_csv(obs_path, sep='\t', comment='#',
                         names=names, header=None, index_col=2)
    dataframe = dataframe[2:]
    dataframe['streamflow'] = dataframe['streamflow'].astype(np.float32)
    dataframe.index = to_datetime(dataframe.index)
    
    #Filter dates
    return dataframe['1960-11-30':'1998-07-30']
    
def get_total(cases, case_id):
    # Take the sum
    total = np.sum(routed_var)
    return total

def plot_sensitivity(cases, group):
    params = cases.sensitivity_params[group]['params']
    dim1 = int( ceil( sqrt( len(params) ) ) )
    dim2 = int( ceil( len(params) / float(dim1) ) )
    fig, axes = plt.subplots(nrows = dim1, ncols=dim2, sharey=False)
    fig.suptitle(
        'Sensitivity Analysis - {}'.format(', '.join(cases.cfg['rvic_fields']) )
    )
    axes_iter = axes.flat
    for param in params:
        case_ids = [case_id for case_id in cases.ratios[group].keys()
                    if case_id.startswith(param)]
        x = [cases.ratios[group][case_id][param] for case_id in case_ids]
        y = [get_total(cases.cases, case_id) for case_id in case_ids]

        subplot = axes_iter.next()
        subplot.scatter(x, y)
        subplot.axis([min(x), max(x), min(y), max(y)])
        subplot.set_title(param)

    fig.tight_layout(rect=(0,0,1,.9))
    dtnow = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    plot_path = 'figures/sensitivity.{datetime}.png'.format(datetime=dtnow)
    plt.savefig(plot_path)
    logging.info('Saving sensitivity analysis plot to {}'.format(plot_path))
    if 'sensitivity' in cases.cfg['plot']:
        plt.show()

def compute_correlation(data, obs):
    return data['streamflow'].corr(obs['streamflow'])

def parallel_plot(data, title):
    df = data.copy()
    cols = list(df)
    x = [i for i, _ in enumerate(cols)]
    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in list(df):
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            ax.plot(x, df.loc[idx, cols], color='green')
        ax.set_xlim([x[i], x[i+1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=20)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]], fontsize=20)


    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])


    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    # Add title
    plt.suptitle(title, fontsize=36)


def compute_absrbias(data, obs):
    sum_obs = sum(obs['streamflow'])
    sum_mod = sum(data['streamflow'])
    return abs((sum_mod - sum_obs) / sum_obs)

def compute_rmse(data,obs):
    n = len(data['streamflow'])
    rmse = np.sqrt(1./ n * np.sum((obs['streamflow'] - data['streamflow'])**2))
    return rmse

def compute_rmseboxcox(data, obs):
    data_bc, _ = stats.boxcox(data['streamflow'])
    obs.loc[obs.streamflow < 0.001, 'streamflow']  = 0.001
    obs_bc, _ = stats.boxcox(obs['streamflow'])
    n = len(data_bc)
    return np.sqrt(1./ n * np.sum((obs_bc - data_bc)**2))

def plot_sed_calibration(directory, result_dir):
    obs = load_obs_data('Observations/OR_discharge.txt')
    obs_plt = obs.reset_index(drop=True)
    cases = [case
             for case in os.listdir(directory)
             if case.startswith('sed_calibrate')
             and os.path.exists(os.path.join(directory, case, result_dir))]

    cols =('id', 'diff')
    stats = dict(zip(cols, ([], [], [], [], [])))
    all_data = DataFrame({'date': date_range('1960-11-01', '1998-06-30')})
    for case in cases:
        case_path = os.path.join(directory, case, result_dir)
        if not os.listdir(case_path):
            continue
        rvic_df = load_rvic_data(case_path = case_path)
        rvic_df_plt = rvic_df.reset_index(drop=True)
        all_data[case[-5:]] = rvic_df_plt['streamflow'].cumsum(axis=0)
        stats['id'].append(case[-5:])
        stats['diff'].append(sum(rvic_df['streamflow']) - 4586)
    stat_df = DataFrame.from_dict(stats)
    stat_df = stat_df.set_index('id')
    print stat_df
    all_data.set_index('date')
    ax = all_data.plot(legend=False, fontsize=20)
    ax.axhline(4586)
    ax.set_xlabel('Date', fontsize=24)
    ax.set_ylabel('Streamflow (cfs)', fontsize=24)
    plt.show()
    
def plot_calibration(directory, result_dir):
    obs = load_obs_data('Observations/OR_discharge.txt')
    obs_plt = obs.reset_index(drop=True)
    cases = [case
             for case in os.listdir(directory)
             if case.startswith('calibrate')
             and os.path.exists(os.path.join(directory, case, result_dir))]
    
    cols =('id', 'Correlation', 'AbsRBias', 'RMSE', 'RMSEboxcox')
    stats = dict(zip(cols, ([], [], [], [], [])))
    for case in cases:
        case_path = os.path.join(directory, case, result_dir)
        if not os.listdir(case_path):
            continue
        rvic_df = load_rvic_data(case_path = case_path)
        rvic_df.index = obs.index
        rvic_df_plt = rvic_df.reset_index(drop=True)
        obs_plt[case[-5:]] = rvic_df_plt[['streamflow']]
        stats['id'].append(case[-5:])
        stats['Correlation'].append(compute_correlation(rvic_df, obs))
        stats['AbsRBias'].append(compute_absrbias(rvic_df, obs))
        stats['RMSE'].append(compute_rmse(rvic_df, obs))
        stats['RMSEboxcox'].append(compute_rmseboxcox(rvic_df, obs))
        
    stat_df = DataFrame.from_dict(stats)
    stat_df = stat_df.set_index('id')
    print stat_df
    obs_plt.index = obs.index
    ax = obs_plt['1970-10-01':'1980-09-30'].plot(
        alpha=0.6, lw=.3, legend=False, fontsize=20)
    obs_plt['1970-10-01':'1980-09-30'][['streamflow']].plot(
        color='black', alpha=1, ax=ax, legend=False)
    ax.set_xlabel('Date', fontsize=24)
    ax.set_ylabel('Streamflow (cfs)', fontsize=24)
    
    parallel_plot(stat_df, '')
    
    # Select the best calibrations to choose from
    stat_df = stat_df.nsmallest(14, 'RMSE')
    stat_df = stat_df.nsmallest(7, 'AbsRBias')
    stat_df = stat_df.nsmallest(5, 'RMSEboxcox')
    print(stat_df)
    
    parallel_plot(stat_df, '')
    print obs_plt.index
    ax2 = obs_plt[stat_df.index.tolist()]['1970-10-01':'1975-09-30'].plot(
        alpha=0.8, lw=.3, fontsize=20)
    obs_plt['1970-10-01':'1975-09-30'][['streamflow']].plot(
        color='black', alpha=1, ax=ax2, legend=False)
    
    plt.show()

    
if __name__ == '__main__':
    plot_sed_calibration('cases/or',
                         os.path.join('rvic',
                                      'dhsvm.convolution', 'hist'))
    
    
