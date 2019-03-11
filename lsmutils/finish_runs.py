from flux2nc_flow import flux2nc
import glob, os, re
import subprocess

d = 'cases/or'
for case in [case for case in glob.glob(d + '/calibrate_*')
             if (os.path.isdir(os.path.join(case, 'vic/output')))
                 and not os.listdir(os.path.join(case, 'rvic/input/forcing'))]:
    vic_out = os.path.join(case, 'vic', 'output')
    idre = re.search(r'calibrate.*', case)
    forcing_fn = idre.group() + '.or.forcing.nc'
    rvic_forcing = os.path.join(case, 'rvic/input/forcing/', forcing_fn)
    convolution_dir = os.path.join(case, 'rvic/config/convolution')
    convolution = os.path.join(convolution_dir, os.listdir(convolution_dir)[0])
    if len(os.listdir(vic_out)) > 200 and not os.listdir(
            os.path.join(case, 'rvic/input/forcing')):
        print case
        flux2nc(vic_out, rvic_forcing, ['OUT_RUNOFF', 'OUT_BASEFLOW'])
        #subprocess.call(['rvic', 'convolution', convolution])
        
