import paramiko

class SSH_Agent():
    def __init__(self, host, username, password):
        self.ssh_fd = paramiko.SSHClient()
        self.ssh_fd.set_missing_host_key_policy( paramiko.AutoAddPolicy() )
        self.ssh_fd.connect( host, username = username, password = password )

    def __del__(self):
        self.ssh_fd.close()

    def run(self, cmd):
        stdin, stdout, stderr = self.ssh_fd.exec_command(cmd)
        return stdout