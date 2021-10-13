import sys
import os
import subprocess
import yaml

params = yaml.safe_load(open("params.yaml"))["correlate"]
getaroundr_path = params["getaroundr_path"]
log_path = params["log_path"]
radius = params["radius"]
cat = params["cat"]
iRA = params["iRA"]
iDEC = params["iDEC"]
asfx = params["asfx"]

for filepath in sys.argv[1:]:
    abspath, basename = os.path.split(filepath)
    out_filepath = os.path.join(abspath, "j_" + basename)
    command = f"{getaroundr_path} -i {filepath} -r {radius} -cat {cat} -o {out_filepath}" \
              f" -asfx {asfx} -iRA {iRA} -iDEC {iDEC} -full"
    stdout_path = os.path.join(log_path, "correlate_log.txt")
    print(command)
    with open(stdout_path, "w") as stdout:
        try:
            subprocess.run(command, stdout=stdout, shell=True, check=True)
        except CalledProcessError as cpe:
            error_msg = "Something went wrong during cross-match step:"
            print(error_msg, cpe)
            raise Exception("Cross-match failed")