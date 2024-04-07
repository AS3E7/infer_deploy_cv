from telnetlib import Telnet

class Telnet_Agent():
    def __init__(self, host, username, password):
        with Telnet(host, 23) as tn:
            tn.interact()

    def __del__(self):
        self.ser.close()

    def run(self, cmd):
        out = self.ser.read(self.ser.in_waiting).decode("gbk")
        return out