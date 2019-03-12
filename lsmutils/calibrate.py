import numpy as np
import logging
import uuid
import glob
import yaml
import datetime
import os

from collections import namedtuple

class Case:

    def __init__(self, cfg, dir_structure):
        self.cfg = cfg
        self.dir_structure = dir_structure
        self.dir_structure.new_case()

    @property
    def input(self):
        input = self.cfg['input'].copy()
        input.update(self.dir_structure.datasets)
        return input

class CaseGroup(yaml.YAMLObject):

    def __init__(self, cfg, inpt):
        self.cfg = cfg
        self.input = inpt

    def initiate_set(self, case_id):
        raise NotImplementedError
    
class BaseCaseGroup(Case, yaml.YAMLObject):

    yaml_tag = '!BaseCase'

    def __init__(self):
        self.cfg = {}

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def initiate_set(self, case_id, cfg):
        self.cfg = cfg.copy()
        self.cfg['case_id'] = case_id
        self.dir_structure = self.cfg['structure'].configure(self.cfg)
        return [Case(self.cfg, self.dir_structure)]
    
class CaseCollection:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cases = []
        for case_id, case_grp in self.cfg['cases'].items():
            self.cases += case_grp.initiate_set(case_id, self.cfg)

class CaseDirStructure(yaml.YAMLObject):

    yaml_tag = '!CaseDirectoryStructure'
    
    def __init__(self, idstr, paths):
        self.idstr = idstr
        self.structure = paths
        self.paths = {}
        self.date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        self._datasets = {}
        
        self.loc = {}
        self.path_info = {}
        layer = [([key], value )
                 for (key, value) in self.structure.items()]
        while layer:
            next_layer = []
            for directory in layer:
                path = directory[0]
                loc = directory[1]
                if loc is None or hasattr(loc, 'file_id'):
                    continue
                self.loc.update({key: path
                                 for key, value in loc.items()
                                 if hasattr(value, 'file_id')})
                self.path_info.update(loc)
                next_layer += [(path + [key], value)
                                for key, value in loc.items()]
            layer = next_layer

    def configure(self, cfg):
        self.paths.update({
            key: self.path_info[key].configure(
                    cfg, file_id=key, dirname=os.path.join(*loc))
            for key, loc in self.loc.items()
        })
        print('Configuring paths')

        # Add static files
        self.paths.update({
            key: path.configure(cfg, file_id=key)
            for key, path in cfg['input'].items()
            if hasattr(path, 'file_id')
        })

        # Generate path strings
        self.files = {key: path.path
                      for (key, path) in self.paths.items() }
        print(self.files)
        return self

    @property
    def datasets(self):
        # Load existing datasets
        if self._datasets:
            self._datasets.update({
                    key: path.dataset 
                    for (key, path) in self.paths.items()
                    if (path.isfile
                        and not key in self._datasets.keys())})
        else:
            self._datasets = { key: path.dataset
                               for (key, path) in self.paths.items()
                               if path.isfile}
        return self._datasets

    def update_datasets(self, new_datasets, prefix=''):
        if prefix:
            new_datasets = {'::'.join([prefix, key]): ds
                            for (key, ds) in new_datasets.items()}
        self.paths.update({key: ds.filepath 
                           for (key, ds) in new_datasets.items()})
        self._datasets.update(new_datasets)
        return self._datasets

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)
        
    def new_case(self):
        for path in self.paths.values():
            if not os.path.exists(path.dirname):
                os.makedirs(path.dirname)
        return self
