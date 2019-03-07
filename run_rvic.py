# API import isn't working
#import rvic
import logging
import subprocess
import os

from path import RVICInputDomainPath, RVICInputRoutingPath
from domain import create_domain_file
from routing import create_routing_file
from operation import *

def run_rvic(cfg, dir_structure):
    logging.info('Starting RVIC parameter generation')
    parameter_path = dir_structure.files['rvic-parameters-cfg']
    logging.info('Config file: {}'.format(parameter_path))
    if not os.path.exists(dir_structure.files['rvic-parameters']):
        subprocess.call(['rvic', 'parameters', parameter_path])

    logging.info('Starting RVIC convolution')
    convolution_path = dir_structure.files['rvic-convolution-cfg']
    logging.info('Config file: {}'.format(convolution_path))
    subprocess.call(['rvic', 'convolution', convolution_path])
