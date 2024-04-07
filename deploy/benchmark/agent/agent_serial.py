import serial

class Serial_Agent():
    def __init__(self, com='/dev/ttyUSB0', bitrate=115200):
        self.ser = serial.Serial(com, bitrate)    # 打开COM17，将波特率配置为115200，其余参数使用默认值
        if self.ser.isOpen():                        # 判断串口是否成功打开
            print("打开串口成功。")
            print(self.ser.name)    # 输出串口号
        else:
            print("打开串口失败。")

    def __del__(self):
        self.ser.close()

    def run(self, cmd):
        out = self.ser.read(self.ser.in_waiting).decode("gbk")
        return out