import copy
import logging
import pkg_resources
import yaml

from .calibrate import CaseCollection
from .operation import Operation
from .utils import write_netcdf

class OperationSequence(yaml.YAMLObject):
    """
    Loads a sequence of GIS operations from a yaml file
    """

    yaml_tag = u'!OpSequence'

    def __init__(self, idstr, name, doc, operations):
        self.idstr = idstr
        self.name = name
        self.doc = doc
        self.operations = operations

        self.computes = [
            output for op in operations for output in op['out'].values()]
        self.requires = [
            req for op in operations for req in op['in'].values()
            if not req in self.computes]
        logging.debug('%s operation requires %s', self.name, self.requires)
        logging.debug('%s operation computes %s', self.name, self.computes)
        

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)
    
    def __repr__(self):
        repr_fmt = ('OperationSequence(name={name}, id={idstr}, ' +
                    'doc={doc}, operations={operations})')
        return repr_fmt.format(name=self.name, idstr=self.idstr,
                               doc=self.doc, operations=self.operations)
    
    @property
    def layers(self):
        next_ids = self.out
        self._layers = []
        while next_ids:
            this_layer = [
                op for op in self.operations
                if any([id in op['out'].values() for id in list(next_ids)])]
            
            # Add layers in reverse order
            self._layers.insert(0, this_layer)
            next_ids = set([idstr for op in this_layer
                                for idstr in op['in'].values()])

        logging.info('Operation sequence layers: \n{}'.format(self._layers))
        return self._layers
        
    def run(self, case, op_in, op_out, seqs):
        logging.info('Running sequence {}'.format(self.idstr))
        
        inpt = op_in.copy()
        inpt.update({op_key: case.input[in_key] 
                      for op_key, in_key in op_in.items()
                      if in_key in case.input})
        
        namespace = {}
        paths = {}
        
        for req, location in inpt.items():
            # Configuration may supply a file to skip required module
            if location in inpt:
                namespace[req] = inpt[location]
                continue
            
            # Run dependency sequences
            logging.debug('Processing dependency %s', location)
            (module, key) = location.split('::')
            if not module in inpt:
                inpt[module] = seqs[module].run(case, input)

            # Build flat namespace
            namespace[req] = input[module][key]

        for lyr in self.layers:
            for op in lyr:
                logging.debug('Preparing to run %s', op['name'])
                
                if op['name'].startswith('seq::'):
                    seq_name = op['name'].replace('seq::', '')
                    case = getattr(seqs, seq_name).run(
                        case, inpt, op['out'], seqs)
                continue
            
                inpt = {key.replace('-', '_'): namespace[value]
                        for key, value in op['in'].items()}
 
                # Optionally configure to save to another location
                for op_key, out_key in op['out'].items():
                    if out_key in case.dir_structure.paths:
                        inpt[op_key] = case.dir_structure.paths[out_key]
                    
                opcls = Operation.__children__()[op['operation']]
                seq_id = list(op_out.values())[0]
                output = opcls(case.cfg, seq_id, **inpt).save()
                namespace.update({out_key: output[op_key]
                                  for out_key, op_key in op['out'].items()})
                
                case.dir_structure.update_datasets(
                    {out_key: namespace[op_key] 
                     for op_key, out_key in op_out.items()
                     if op_key in namespace})
        
        return case

        return case

class OpSeqLoader(object):

    loc = 'sequences'
    
    def __init__(self):
        self.load()

    def load(self):
        for resource in pkg_resources.resource_listdir(__name__, self.loc):
            if resource.endswith('.yaml'):
                logging.debug('Loading %s', resource)
                res_path = '/'.join([self.loc, resource])
                opseq_def = pkg_resources.resource_string(__name__, res_path)
                opseq = yaml.load(opseq_def)
                setattr(self, opseq.idstr, opseq)

    def __getitem__(self, key):
        return getattr(self, key)

def run_cfg(cfg):
    collection = CaseCollection(cfg)
    cases = collection.cases
    case = cases[0]

    seqs = OpSeqLoader()
    
    master_seq = OperationSequence(
        'main',
        'Main Sequence',
        '',
        cfg['operations']
    )

    logging.info('Operations to run:')
    for op in master_seq.operations:
        logging.info('  %s', op['name'])
        for key, value in op['in'].items():
            logging.info('    I: %s <- %s', key, value)
        for key, value in op['out'].items():
            logging.info('    O: %s <- %s', key, value)

    case = master_seq.run(
            case,
            {key: key for key in master_seq.requires},
            {key: key for key in master_seq.computes},
            seqs
    )
    return case
