from osgeo import gdal
import logging
import os, sys
from multiprocessing import Pool

from path import RVICInputTempPath as TempPath
from run_rvic import run_rvic
from run_vic import run_vic
from calibrate import CaseCollection
from sequence import seqs

def run_case(case):
    # Write new files matching case parameters
    case.write_file()

    # Delineate watershed
    case.dir_structure.datasets.update(seqs.delineate.run(case))
    boundary_ds = case.dir_structure.datasets['boundary']

    # Run VIC
    run_vic(case.cfg, case.dir_structure, boundary_ds)

    # Run RVIC
    case.dir_structure.datasets.update(seqs.domain.run(case))
    case.dir_structure.datasets.update(seqs.routing.run(case))
    run_rvic(case.cfg, case.dir_structure)

    # Remove unnecessary files
    if os.path.exists(case.dir_structure.files['vic-result']):
        shutil.rmtree(case.dir_structure.files['vic-result'])


if __name__ == '__main__':
    # Load configuration file
    cfg = yaml.load(open('config/test.yaml', 'r').read())
    logging.basicConfig(stream=sys.stdout, level=cfg['log_level'])
    logging.debug(cfg)
    cfg['field_str'] = '.'.join(cfg['rvic_fields'])
    
    # Compute VIC start and stop dates from RVIC and spin-up
    cfg['vic_start'] = dict(
        zip(['year', 'month', 'day', 'hour'],
                map(int, cfg['run_start_date'].split('-') )
                )
        )
    # Subtract spin-up years
    cfg['vic_start']['year'] -= int(cfg['vic_spin_up'])
    cfg['vic_stop'] = dict(
        zip(['year', 'month', 'day', 'hour'],
                map(int, cfg['run_stop_date'].split('-'))
                )
        )
    # Add a month - RVIC needs some extra
    cfg['vic_stop']['month'] = cfg['vic_stop']['month'] % 12 + 1
    gdal.UseExceptions()
    
    # Clear and create directory for result files
    temp_path = TempPath('', **cfg)
    if os.path.exists(temp_path.dirname):
        shutil.rmtree(temp_path.dirname)
    
    # Generate case configurations for 
    collection = CaseCollection(cfg)
    cases = collection.cases
    logging.debug(cases)

    #pool = Pool(processes=1)    
    #pool.map(run_case, cases)
    for case in cases:
        run_case(case)

    """  'sensitivity_params': {
            'soil-params': {
                'params': [
                    'ds',
                    'dsmax',
                    'ws',
                    'C',
                    'ksat_1',
                    'ksat_2',
                    'phi_s_1',
                    'phi_s_2',
                    'depth_1',
                    'depth_2',
                    'expt_1',
                    'expt_2',
                    'b_infilt',
                    'k_index', 
                    'd50',
                    'soil_cohesion',
                    'critical_area'
                ],
                'ratios': [0, 1/3., 2/3., 1],
            }
        }
                
        'calibration_params': {
            'params': {
                'soil-params': [
                    'depth_2',
                    'd50'
                ],
            },
            'ninterval': 10
        },
       'forcing_params': {
            'precipitation': [0.7, 0.9, 1.1, 1.3],
            'tmax': [-2,-1, 1, 2]
        }"""

    
        

