import argparse
import logging
import sys
import yaml

from lsmutils.sequence import run_cfg
import lsmutils.operation.operation
from lsmutils.sequence import *

if __name__=='__main__':

    parser = argparse.ArgumentParser(
            description='Run modeling utilities as configured')
    parser.add_argument('cfg_path')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    cfg = yaml.load(open(args.cfg_path, 'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.info('Running configuration file: %s', args.cfg_path)

    run_cfg(cfg)
