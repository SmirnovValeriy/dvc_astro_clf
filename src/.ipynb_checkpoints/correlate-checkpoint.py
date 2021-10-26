import os
import subprocess
import yaml

params = yaml.safe_load(open('params.yaml'))
params_common = params['common']
log_path = params_common['log_path']
params = params['correlate']
getaroundr_path = params['getaroundr_path']
radius = params['radius']
cat = params['cat']
iRA = params['iRA']
iDEC = params['iDEC']
asfx = params['asfx']
postfixes = ['01', '02', '11', '12']

for postfix in postfixes:
    filepath = 'data/cv' + postfix + '.fits'
    out_filepath = 'data/j_cv' + postfix + '.fits'
    command = f"{getaroundr_path} -i {filepath} -r {radius} -cat {cat}"  \
        f" -o {out_filepath} -asfx {asfx} -iRA {iRA} -iDEC {iDEC} -full"
    stdout_path = os.path.join(log_path, 'correlate_log.txt')
    print(command)
    with open(stdout_path, 'w') as stdout:
        try:
            subprocess.run(command, stdout=stdout, shell=True, check=True)
        except subprocess.CalledProcessError as cpe:
            error_msg = 'Something went wrong during cross-match step:'
            print(error_msg, cpe)
            raise Exception('Cross-match failed')
