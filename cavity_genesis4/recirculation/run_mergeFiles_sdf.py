import numpy as np
import pickle
import os
import subprocess
from time import sleep
from genesis4.lume_genesis_JT.tools import all_done


def submit_mergeFiles(shell_script = 'RunMergeFiles.sh'):
    cmd = 'sbatch '+ shell_script
    print(cmd)
    x = subprocess.check_output(cmd.split())
    y = x.split()
    print(x)
    return y[-1]



def start_mergeFiles(nRoundtrips, workdir, saveFilenamePrefix, dgrid, dt, Dpadt):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/merge_params.p", "wb" ) )
    
    os.system('cp  cavity_genesis4/recirculation/RunMergeFiles.sh ' + workdir)
    os.system('cp  cavity_genesis4/recirculation/merge_files_mpi.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_mergeFiles()
    os.chdir(root_dir)
    all_done([jobid])
    
    return jobid
