import netCDF4 as nc4
import logging
import datetime
import yaml
import numpy as np

def write_netcdf(out_uri, var_dict, fill=-99):
    ncfile = nc4.Dataset(out_uri, 'w', format='NETCDF4_CLASSIC')
    ncfile.history = 'Created using rvic_prep.py {}'.format(
            datetime.datetime.now())
    
    dims_created = False
    for var, [ds, dtype, units] in var_dict.iteritems():
        array = ds.array
        nodata = ds.nodata
        array[array==nodata] = fill
        
        if not dims_created:
            t_dim = ncfile.createDimension("time", None)
            t_var = ncfile.createVariable("time", "f8", ("time",))
            t_var.units = 'time'
            t_var[:] = np.array([0])
            
            lons = ds.cgrid.lon
            lon_dim = ncfile.createDimension("lon", len(lons))
            lon_var = ncfile.createVariable("lon", "f8", ("lon",))
            lon_var[:] = lons
            lon_var.units = 'degrees_east'
            
            lats = ds.cgrid.lat
            lat_dim = ncfile.createDimension("lat", len(lats))
            lat_var = ncfile.createVariable("lat", "f8", ("lat",))
            lat_var[:] = lats
            lat_var.units = 'degrees_north'

            #fill_dim = ncfile.createDimension('_FillValue', 1)
            #fill_var = ncfile.createVariable('_FillValue', 'i', ('_FillValue',))
            #fill_var = fill

            dims_created = True

        ncvar = ncfile.createVariable(var, dtype, ('time', 'lat', 'lon'))
        ncvar.missing_value = fill
        ncvar.units = units
        ncvar[0:1,:,:] = np.expand_dims(array, axis=0)
        
    ncfile.close()
    logging.info('Netcdf file written to {}'.format(out_uri))

class CoordProperty(yaml.YAMLObject):
    
    yaml_tag = u"!Coord"

    def __init__(self, x=None, y=None, lon=None, lat=None, epsg=4326):
        if x is None:
            self._x = lon
        else:
            self._x = x

        if y is None:
            self._y = lat
        else:
            self._y = y

        # Get correct datatypes when loading from yaml
        try:
            self._x = eval(self._x)
            self._y = eval(self._y)
        except:
            pass

        self.epsg = epsg

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def lon(self):
        return self._x

    @property
    def lat(self):
        return self._y

    @property
    def coord(self):
        return (self.x, self.y)

    def grid_coord(self, res, method=round):
        new_x = method(self.x / float(res.x)) * res.x
        new_y = method(self.y / float(res.y)) * res.y
        return CoordProperty(x=new_x, y=new_y)

    def __add__(self, other):
        return CoordProperty(x = (self.x + other.x), y = (self.y + other.y))
    
    def __sub__(self, other):
        return CoordProperty(x = (self.x - other.x), y = (self.y - other.y))

    def __div__(self, other):
        try:
            return CoordProperty(x = self.x / other.x, y = self.y / other.y)
        except AttributeError:
            return CoordProperty(x = self.x / other, y = self.y / other)

    def __truediv__(self, other):
        try:
            return CoordProperty(x = self.x / other.x, y = self.y / other.y)
        except AttributeError:
            return CoordProperty(x = self.x / other, y = self.y / other)

    def __format__(self, format_spec):
        return '{}_{}'.format(self.x, self.y)

class BBox(yaml.YAMLObject):
    
    yaml_tag = u"!BBox"

    def __init__(self, llc, urc):
        self.llc = llc
        self.urc = urc

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)
