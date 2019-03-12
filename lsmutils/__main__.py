import sys
import argparse
import logging
import yaml

from .sequence import run_cfg

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Run modeling utilities as configured')
    parser.add_argument('cfg_path')
    args = parser.parse_args()

    cfg = yaml.load(open(args.cfg_path, 'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.info('Running configuration file: %s', args.cfg_path)
    run_cfg(cfg)
