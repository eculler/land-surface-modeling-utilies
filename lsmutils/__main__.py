import pkg_resources
import yaml
import argparse
from sequence import *
from run_cases import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prepare and run some models')
    
    gdal.UseExceptions()
    # Clear and create directory for result files
    temp_path = TempPath('', **cfg)
    if os.path.exists(temp_path.dirname):
        shutil.rmtree(temp_path.dirname)

    # Generate case configurations for 
    collection = CaseCollection(config)
    cases = collection.cases
    case = cases[0]
