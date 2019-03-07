#!/usr/bin/python

#----------------------------------------------------
# Program to convert VIC fluxes files to NetCDF file
# will ask the user which variable he wants to export
# and also for which years. Assumes there is data
# for the entire time period, from 1-jan to 31-dec
# SET UP FOR DAILY TIME STEP. FLUX FILE SHOULD NOT
# CONTAIN HOUR RECORD!!
#----------------------------------------------------

#------------------------------------------------
# Writen by Daniel de Castro Victoria
# dvictori@cena.usp.br or daniel.victoria@gmail.com
# Needs python libraries Numeric and Scientific
# 03-dec-2004
# Edited by Jenna Stewart
# jenna.r.stewart@colorado.edu
# 02-feb-2016: updates to netCDF packages
# 18-mar-2016: updates to date management to allow for
# random VIC run dates
# Edited by Elsa Culler
# elsa.culler@colorado.edu
# 13-feb-2018: clean up and eliminated need to edit script before running
#-------------------------------------------------

import os, sys, string, re, csv
# handle dates...
from datetime import datetime
# NetCDF and Numeric
from scipy.io import netcdf
from numpy import *


def flux2nc(flux_dir, out_path, out_vars):
    # building file list and sorted lat lon list
    file_list = os.listdir(flux_dir)

    # Pull lat and lon out of filenames
    fn_re_str = r'[^.]\w*_(?P<lat>-?\d{1,3}.\d{5})_(?P<lon>-?\d{1,3}.\d{5})'
    filename_re = re.compile(fn_re_str)
    lats = []
    lons = []
    for filename in file_list:
        match = filename_re.match(filename)
        if match:
            if not match.group('lat') in lats: lats.append(match.group('lat')) 
            if not match.group('lon') in lons: lons.append(match.group('lon')) 

    # putting in order. Lat should be from top to botom
    # lon from left to right
    lons.sort(reverse=True)
    lats.sort(reverse=True)

    # Extract dates and header
    with open(os.path.join(flux_dir, file_list[0]), "r") as flux_file:
        reader = csv.reader(flux_file, delimiter='\t')
        # Extract the start date...
        first = next(reader)
        while first[0].startswith('#'):
            header = first
            first = next(reader)
            
        # the start date
        start_dt = datetime(int(first[0]), int(first[1]), int(first[2]))

        # the header...
        header = [col.split(' ')[-1] for col in header]

        # and the end date
        for last in reader:
            pass
        end_dt = datetime(int(last[0]), int(last[1]), int(last[2]))

    # Need the number of records to preallocate array
    days = (end_dt - start_dt).days + 1


    print("Go grab a coffee, this could take a while...")
    # Create an array with -9999 (NoData)
    # Then populate the array by reading each flux file
    data = {var: ones([days, len(lats), len(lons)]) * -9999 for var in out_vars}
    files_remaining = len(file_list)
    for filename in file_list:
        print("Processing {}".format(filename))
        match = filename_re.match(filename)
        # Don't read files that don't match the flux file pattern
        if not match:
            continue
        lat_ind = lats.index(match.group('lat'))
        lon_ind = lons.index(match.group('lon'))

        with open(os.path.join(flux_dir, filename), "r") as flux_file:
            file_data = {var: [] for var in out_vars}
            # Lines starting with # are commented and contain no data
            reader = csv.DictReader(filter(lambda row: row[0]!='#', flux_file),
                                    fieldnames=header, delimiter='\t')
            for row in reader:
                if list(row.values())[0].startswith('#'):
                    continue
                for var in out_vars:
                    file_data[var].append(row[var])
                    # Save to array
            for var in out_vars:
                data[var][:, lat_ind, lon_ind] = file_data[var]

        # Status update - this can take awhile
        print('{} files remaining'.format(files_remaining))
        files_remaining -= 1


    #------------------------------------------#
    # writing NetCDF
    #------------------------------------------#

    with netcdf.netcdf_file(out_path, "w") as ncfile:

        ncfile.Conventions = "COARDS"
        ncfile.history = "Created using flux2nc.py. " + str(datetime.now())
        ncfile.production = "VIC output"

        ncfile.start_date = start_dt.isoformat()
        ncfile.end_date = start_dt.isoformat()

        #create dimensions
        ncfile.createDimension("lon", len(lons))
        ncfile.createDimension("lat", len(lats))
        ncfile.createDimension("time", days)

        #create variables
        latvar = ncfile.createVariable("lat", 'f8', ("lat",))
        latvar.long_name = "Latitude"
        latvar.units = "degrees_north"
        latvar[:] = lats

        lonvar = ncfile.createVariable("lon", 'f8', ("lon",))
        lonvar.long_name = "Longitude"
        lonvar.units = "degrees_east"
        lonvar[:] = lons

        timevar = ncfile.createVariable("time", 'f8', ("time",))
        timevar.long_name = "time"
        timevar.units = "days since " + datetime(1,1,1).isoformat()
        timevar.calendar = "standard"
        timevar[:] = range((start_dt - datetime(1,1,1)).days,
                           (end_dt - datetime(1,1,1)).days + 1)

        for var in out_vars:
            var_str = var.replace('OUT_', '').replace('SED_', '')
            data_var = ncfile.createVariable(
                var_str.lower(), 'f8', ("time","lat","lon"))
            data_var.long_name =  "{} calculated by VIC".format(var_str.title())
            data_var.missing_value = -9999.0
            data_var.units = "mm"
            data_var[:] = data[var]

        ncfile.close()

    print("python script complete")

if __name__ == '__main__':
    # checking user input
    if len(sys.argv) != 5:
        print("Wrong user input")
        print("Convert VIC fluxes files to NetCDF")
        print("usage flux2nc_flow.py <vic flux dir> <output directory> <output prefix> <output variable 1>,<output variable 2>")
        sys.exit()

    flux_dir = sys.argv[1]
    out_dir = sys.argv[2]
    out_prefix = sys.argv[3]
    out_vars = sys.argv[4].split(',')

    flux2nc(flux_dir, out_dir, out_prefix, out_vars)
