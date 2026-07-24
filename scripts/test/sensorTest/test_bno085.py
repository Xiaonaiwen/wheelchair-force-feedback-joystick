import time
import math
import board
import busio
from adafruit_bno08x import BNO_REPORT_LINEAR_ACCELERATION, BNO_REPORT_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c)

bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

def quat_to_euler(qi, qj, qk, qr):
    # roll
    sinr_cosp = 2 * (qr * qi + qj * qk)
    cosr_cosp = 1 - 2 * (qi * qi + qj * qj)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    # pitch
    sinp = 2 * (qr * qj - qk * qi)
    if abs(sinp) >= 1:
        pitch = math.degrees(math.copysign(math.pi / 2, sinp))
    else:
        pitch = math.degrees(math.asin(sinp))

    # yaw
    siny_cosp = 2 * (qr * qk + qi * qj)
    cosy_cosp = 1 - 2 * (qj * qj + qk * qk)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw

while True:
    ax, ay, az = bno.linear_acceleration
    qi, qj, qk, qr = bno.quaternion
    roll, pitch, yaw = quat_to_euler(qi, qj, qk, qr)

    # print("linear accel:", ax, ay, az)
    print("angle:", roll, pitch, yaw)
    print()
    time.sleep(1)

#at ay negative means positive
#roll is one we need in degree