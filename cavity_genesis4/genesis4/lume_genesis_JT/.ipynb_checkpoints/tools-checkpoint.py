from time import sleep
import subprocess
import os


def submit_genesis(input_file='HXR.in', shell_script = 'RunGenesis2.sh'):
    cwd = os.getcwd()
    print(cwd)
    cmd = 'sbatch '+ shell_script + ' ' + input_file
    print(cmd)
    x = subprocess.check_output(cmd.split())
    y = x.split()
    print(x)
    return y[-1]

def all_done(jid):
    flag = [False for i in range(len(jid))]
    all_done_flag = False
    while not all_done_flag:
        sleep(10)
        count = 0
        for id in jid:
            ret2=subprocess.getoutput("squeue -u $(whoami)")
            if ret2.count(str(int(id))) < 1:
                flag[count] = True
            count +=1
        all_done_flag = all(flag)
        print("job "  + str(jid[0]) + " is running")
    print('all done!')