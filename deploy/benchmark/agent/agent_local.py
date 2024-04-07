import subprocess
import os

class Local_Agent():
    def __init__(self, host, username, password):
        pass

    def __del__(self):
        pass

    def run(self, cmd):
        pi = subprocess.call(cmd, env=os.environ, shell=True, stdout=subprocess.PIPE)
        for i in iter(pi.stdout.readline, 'b'):
            print(i)
        # return stdin, stdout, stderr