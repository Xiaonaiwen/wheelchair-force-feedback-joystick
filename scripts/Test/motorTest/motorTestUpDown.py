from src.motor.motor import *
import time

upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()

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
    upDownMotor.pidForConstantPosition(upDownMotor.startPosition_30, Kp = 0.05, Ki = 0.05, Kd = 0)
