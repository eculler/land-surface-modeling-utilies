import copy
import logging
import os
import pkg_resources
import random
import shutil
import string
import yaml

from .calibrate import CaseCollection
from .operation import Operation


class OperationSequence(yaml.YAMLObject):
    """
    Loads a sequence of GIS operations from a yaml file
    """

    yaml_tag = u'!OpSequence'

    def __init__(
            self, operations, name='cfg', title='Configuration File', doc=''):
        self.name = name
        self.title = title
        self.doc = doc
        self.inpt = {}
        self.out = {}
        self.id = ''.join([random.choice(string.ascii_letters + string.digits)
                           for n in range(6)])

        self._new_labels = None

        # Unpack subsequences
        self.operations = []
        new_labels = {}
        for step in operations:
            if hasattr(step, 'operations'):
                new_labels.update(step.new_labels)
                self.operations.extend(step.operations)
            else:
                logging.debug('New labels:')
                for key, value in new_labels.items():
                    logging.debug('    %s <- %s', key, value)
                step.relabel(new_labels)
                self.operations.append(step)

        logging.debug('%s operation computes %s', self.name, self.computes)

    @classmethod
    def from_yaml(cls, loader, node):
        seq_class = cls
        fields = loader.construct_mapping(node, deep=True)

        if not 'operations' in fields:
            seq_name = fields['name'].replace('-', '_') + '.yaml'
            seq_path = '/'.join(['sequences', seq_name])
            seq_def = pkg_resources.resource_string(__name__, seq_path)
            seq = yaml.load(seq_def)
            seq.configure(fields['in'], fields['out'])
            return seq

        return cls(**fields)

    def configure(self, inpt, out):
        self.inpt = inpt
        self.out = out
        for op in self.operations:
            op.relabel(self.new_labels)

    def __repr__(self):
        repr_fmt = ('OperationSequence(name={name}, ' +
                    'doc={doc}, operations={operations})')
        return repr_fmt.format(
                name=self.name,
                doc=self.doc,
                operations=[op.name for op in self.operations])

    @property
    def computes(self):
        return [
            output for op in self.operations
            for output in op.out.values()
        ]

    @property
    def new_labels(self):
        if not self._new_labels:
            print(self.out)
            self._new_labels = {
                value: key for key, value in self.out.items()}
            self._new_labels.update({
                output: '{}_{}'.format(output, self.id)
                for op in self.operations
                for output in op.out.values()
                if not output in self.out
            })
            self._new_labels.update(self.inpt)
        return self._new_labels

    @property
    def requires(self):
        return [
            inpt for op in self.operations
            for inpt in op.inpt.values()
            if not inpt in self.computes
        ]

    def run(self, case):
        logging.info('Running {} sequence'.format(self.title))

        for op in self.operations:
            inpt_data = copy.deepcopy(op.inpt)

            # Get datasets from the case, if data exists already
            inpt_data.update({
                key: case.dir_structure.data[value]
                for key, value in inpt_data.items()
                if value in case.dir_structure.data
            })

            if inpt_data:
                logging.debug('Input files located at:')
            for key, ds in inpt_data.items():
                if hasattr(ds, 'loc'):
                    logging.debug('    %s <- %s', key, ds.loc.path)

            if case.dir_structure.output_files:
                logging.debug('Output files located at:')
            for key, ds in case.dir_structure.output_files.items():
                if hasattr(ds, 'loc'):
                    logging.debug('    %s <- %s', key, ds.loc.path)

            # Run operation
            output_data = op.configure(
                    case.cfg,
                    locs=case.dir_structure.output_files,
                    scripts=case.dir_structure.scripts,
                    **inpt_data).save()

            logging.debug('Output files saved to:')
            for key, ds in output_data.items():
                if hasattr(ds, 'loc'):
                    logging.debug('    %s <- %s', key, ds.loc.path)

            new_data = {
                out_key: output_data[op_key]
                for op_key, out_key in op.out.items()
                if op_key in output_data
            }

            logging.debug('Files added to case:')
            for key, ds in new_data.items():
                if hasattr(ds, 'loc'):
                    logging.debug('    %s <- %s', key, ds.loc.path)

            case.dir_structure.update_data(new_data)

        return case

def run_cfg(cfg):
    #logging.debug('Loaded configuration \n%s', yaml.dump(cfg))

    collection = CaseCollection(cfg)
    cases = collection.cases
    case = cases[0]

    master_seq = OperationSequence(cfg['operations'])
    master_seq.configure(
        inpt={key: key for key in list(cfg['in'].keys())},
        out=case.dir_structure.output_files)

    logging.info('Operations to run:')
    for op in master_seq.operations:
        logging.info('  %s', op.title)
        for key, value in op.inpt.items():
            logging.info('    I: %s <- %s', key, value)
        for key, value in op.out.items():
            logging.info('    O: %s <- %s', key, value)

    case = master_seq.run(case)

    # Clean up
    if cfg['log_level'] > logging.DEBUG:
        tmp_path = os.path.join(cfg['base_dir'], cfg['temp_dir'])
        shutil.rmtree(tmp_path)

    return case
