from agent.agent_local import *
from agent.agent_ssh import *
from agent.agent_serial import *
from agent.agent_telnet import *
from agent.agent_adb import *

def get_agent(config):
    if config['type'] == 'ssh':
        return SSH_Agent(config['ip'], config['username'], config['password'])
    elif config['type'] == 'serial':
        return Serial_Agent(config['com'], config['bitrate'])
    elif config['type'] == 'telnet':
        return Telnet_Agent(config['ip'], config['username'], config['password'])
    elif config['type'] == 'adb':
        return Adb_Agent(config['ip'])

    return Local_Agent()