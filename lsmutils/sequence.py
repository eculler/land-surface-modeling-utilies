import copy
import logging
import os
import pkg_resources
import random
import shutil
import string
import yaml

from lsmutils.calibrate import CaseCollection
import lsmutils.operation


class OperationSequence(yaml.YAMLObject):
    """
    Loads a sequence of GIS operations from a yaml file
    """

    yaml_tag = '!OpSequence'

    def __init__(
            self, operations, name='cfg', title='Configuration File', doc=''):
        self.name = name
        self.title = title
        self.doc = doc
        self.inpt = {}
        self.out = {}
        self.id = ''.join([random.choice(string.ascii_letters + string.digits)
                           for n in range(6)])

        # Unpack subsequences
        self.operations = operations

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
            dims = fields['dims'] if 'dims' in fields else []
            seq.configure(fields['in'], fields['out'], dims)
            return seq

        return cls(**fields)

    def configure(self, inpt, out, dims=[]):
        self.inpt = inpt
        self.out = out
        self.dims = dims

        new_ops = []
        intermediate = []

        for step in self.operations:
            # Tag intermediate operation output
            for key, label in step.out.items():
                if str(label) not in list(self.out) + list(self.inpt):
                    intermediate.append(label)
                    step.out[key] = '{}_{}'.format(label, self.id)

        for step in self.operations:
            # Tag intermediate operation input
            for key, label in step.inpt.items():
                if str(label) in intermediate:
                    step.inpt[key] = '{}_{}'.format(label, self.id)

        for step in self.operations:
            # Relabel operations next layer down
            if hasattr(step, 'operations'):
                new_labels = step.inpt.copy()
                new_labels.update(step.out)
                for op in step.operations:
                    op.relabel(new_labels)
                    op.dims = list(set(op.dims).union(set(step.dims)))
                # Expand subsequence
                new_ops.extend(step.operations)
            else:
                new_ops.append(step)
            self.operations = new_ops

    def relabel(self, new_labels):
        for key, dsname in self.inpt.items():
            if str(dsname) in new_labels:
                self.inpt[key] = new_labels[dsname]
                logging.debug(
                    'Relabelled %s to %s', dsname, new_labels[dsname])

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
    def requires(self):
        return [
            inpt for op in self.operations
            for inpt in op.inpt.values()
            if not inpt in self.computes
        ]

    def run(self, case):
        logging.info('Running {} sequence'.format(self.title))

        if case.dir_structure.output_files:
            logging.debug('Output files located at:')
        for key, loc in case.dir_structure.output_files.items():
            if hasattr(loc, 'path'):
                logging.debug('    %s\n    %s', key, loc)

        for op in self.operations:
            inpt_data = copy.deepcopy(op.inpt)

            # Get hardcoded data from the case
            inpt_data.update({
                key: case.dir_structure.data[value]
                for key, value in inpt_data.items()
                if str(value) in case.dir_structure.data
            })

            # Apply configured file names
            if inpt_data:
                logging.debug('Input files located at:')
                for key, loc in inpt_data.items():
                    if hasattr(loc, 'file_id'):
                        loc.remove_missing()
                    if hasattr(loc, 'path'):
                        logging.debug('    %s\n    %s', key, loc)

            # Run operation
            output_locs = op.configure(
                case.cfg,
                locs=case.dir_structure.output_files,
                scripts=case.dir_structure.scripts,
                **inpt_data).save()

            logging.debug('Output files saved to:')
            for key, loc in output_locs.items():
                if hasattr(loc, 'path'):
                    logging.debug('    %s\n    %s', key, loc)

            new_locs = {
                out_key: output_locs[op_key]
                for op_key, out_key in op.out.items()
                if op_key in output_locs
            }

            logging.debug('Files added to case:')
            for key, ds in new_locs.items():
                if hasattr(ds, 'loc'):
                    logging.debug('    %s\n    %s', key, ds.loc)

            case.dir_structure.update(new_locs)

        return case

def run_cfg(cfg):
    logging.debug('Loaded configuration \n%s', yaml.dump(cfg))

    collection = CaseCollection(cfg)
    cases = collection.cases
    case = cases[0]

    cfg['sequence'].configure(
        inpt={key: key for key in list(cfg['in'].keys())},
        out=case.dir_structure.output_files)

    logging.info('Operations to run:')
    for op in cfg['sequence'].operations:
        logging.info('  %s', op.title)
        for key, value in op.inpt.items():
            logging.info('    I: %s <- %s', key, value)
        for key, value in op.out.items():
            logging.info('    O: %s <- %s', key, value)

    tmp_path = os.path.join(cfg['base_dir'], cfg['temp_dir'])
    try:
        shutil.rmtree(tmp_path)
    except:
        pass

    case = cfg['sequence'].run(case)

    # Clean up
    if cfg['log_level'] > logging.DEBUG:
        try:
            shutil.rmtree(tmp_path)
        except:
            pass

    return case
