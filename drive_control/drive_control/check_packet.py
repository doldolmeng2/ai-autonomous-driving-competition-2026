import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)

for i in range(20):
    packet = '-30 20\n'
    print(f'[TX] {packet.strip()}')
    ser.write(packet.encode())
    ser.flush()

    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f'[ARDUINO] {line}')

    time.sleep(0.1)

ser.write(b'0 0\n')
ser.flush()
ser.close()