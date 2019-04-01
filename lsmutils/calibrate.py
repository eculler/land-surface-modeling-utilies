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
        input = self.cfg['in'].copy()
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

Directory = namedtuple('Directory', ['path', 'contents'])

class CaseDirStructure(yaml.YAMLObject):

    yaml_tag = '!CaseDirectoryStructure'

    def __init__(self, idstr, paths):
        self.idstr = idstr
        self.structure = paths
        self.locs = {}
        self.date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.data = {}

        self.path_info = {}
        self.output_files = {}
        layer = [Directory(path=[name], contents=contents)
                 for (name, contents) in self.structure.items()]

        while layer:
            next_layer = []
            for directory in layer:
                # Stop iteration
                if (hasattr(directory.contents, 'file_id') or
                        directory.contents is None):
                    continue

                # Save location of each output file
                self.output_files.update({
                    key: directory.path
                    for key, value in directory.contents.items()
                    if hasattr(value, 'file_id')})

                # Save contents of each directory
                self.path_info.update(directory.contents)

                # Include subdirectories in the next layer
                next_layer += [
                    Directory(directory.path + [key], value)
                    for key, value in directory.contents.items()]

            layer = next_layer

    def configure(self, cfg):
        # Configure locators
        # Add output files
        self.output_files = {
            key: self.path_info[key].configure(
                cfg, file_id=key, dirname=os.path.join(*loc))
            for key, loc in self.output_files.items()
        }
        self.locs.update(self.output_files)
        # Add input files
        self.locs.update({
            key: path.configure(cfg, file_id=key)
            for key, path in cfg['in'].items()
            if hasattr(path, 'file_id')
        })

        # Configure datasets
        # Start with non-file data
        self.data.update(cfg['in'])
        # Add existing datasets
        self.data.update({
            key: loc.dataset
            for (key, loc) in self.locs.items()
            if loc.isfile
        })

        # Configure script locations:
        self.scripts = {}
        if 'scripts' in cfg:
            self.scripts.update(
                {key: loc.path for key, loc in cfg['scripts'].items()})

        # Report input and output files
        for key, loc in self.locs.items():
            filetype = 'Output' if key in self.output_files else 'Input'
            logging.info('%s file %s located at %s', filetype, key, loc.path)

        return self

    def update(self, new_data):
        self.locs.update(new_data)
        self.data.update({
            key: loc.dataset
            for (key, loc) in self.locs.items()
            if loc.isfile
        })
        return self.data

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def new_case(self):
        for path in self.locs.values():
            if not os.path.exists(path.dirname):
                os.makedirs(path.dirname)
        return self
