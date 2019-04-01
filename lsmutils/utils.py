import netCDF4 as nc4
import logging
import datetime
import yaml
import numpy as np

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
        self._llc = llc
        self._urc = urc

    @property
    def llc(self):
        return self._llc

    @property
    def urc(self):
        return self._urc

    @property
    def min(self):
        return self._llc

    @property
    def max(self):
        return self._urc
        
    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)
