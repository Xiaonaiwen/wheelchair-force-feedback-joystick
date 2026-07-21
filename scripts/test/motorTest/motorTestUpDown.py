from src.motor.motor import *
import time

upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()

upDownMotor.setToInitialPosition()
leftRightMotor.setToInitialPosition()
time.sleep(1)
upDownMotor.changeMode()

"""
while True:
    pos, vec = upDownMotor.detectPositionVelocity()
    vel_cmd, acc_cmd = upDownMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
    print("acc cmd: " + str(acc_cmd))
    time.sleep(1)
"""

while True:
    upDownMotor.pidForConstantPosition(upDownMotor.startPosition_30, Kp = 1.25, Ki = 0, Kd = 0.07)
    pos, vec = upDownMotor.detectPositionVelocity()
    vel_cmd, _ = upDownMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
 