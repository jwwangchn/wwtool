import array
import binascii
import serial
import serial.tools.list_ports

class UART():
    def __init__(self, 
                port_name="/dev/ttyUSB0", 
                baud_rate=9600,
                time_out=5):
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.time_out = time_out

    def available_port(self):
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)
        if len(port_list) <= 0:
            print("Not available port")
            return False
        else:
            port_list_0 = list(port_list[0])
            serial_name = port_list_0[0]
            print("Available port: ", serial_name)
            return True

    def open_serial(self):
        self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=self.time_out)
    
    def serial_state(self):
        print("Serial port name: {}".format(self.ser.name))
        print("Serial read timeout: {}s".format(self.ser.timeout))
    
    def send_msg(self, msg, mode=1, debug=True):
        """
        mode=1 -> 10
        mode=2 -> hex, for example: [0xaa, 0x12]
        """
        if debug:
            print("Sending: {}".format(msg))
        if mode == 1:
            self.ser.write(msg)
        if mode == 2:
            msg = array.array("B", msg).tostring()
            self.ser.write(msg)        

    def receive_msg(self, buffer=2, mode=1, hex_mode=False, debug=True):
        """
        mode=1 -> read buffer
        mode=2 -> read lines
        hex_mode=True -> return hex message
        """
        if mode == 1:
            msg = self.ser.read(buffer)
        if mode == 2:
            msg = self.ser.readline()
        if hex_mode:
            msg = binascii.hexlify(msg).decode("utf-8")

        if debug:
            print("Receive: {}".format(msg))
        return msg

    def hexstr2int(self, hexstr):
        return [int(b, 16) for b in [hexstr[i:i+2] for i in range(0, len(hexstr), 2)]]

    def close_serial(self):
        self.ser.close()



    
