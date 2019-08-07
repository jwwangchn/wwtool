from wwtool import UART


if __name__ == '__main__':
    ser = UART(port_name="/dev/ttyUSB0", 
                baud_rate=9600,
                time_out=1)

    ser_state = ser.available_port()
    
    if ser_state:
        ser.open_serial()
        ser.serial_state()
        while True:
            # uart.send_msg([0xaa, 0x12], mode=2)
            ser.receive_msg(buffer=2, mode=1, hex_mode=True)

    ser.close_serial()