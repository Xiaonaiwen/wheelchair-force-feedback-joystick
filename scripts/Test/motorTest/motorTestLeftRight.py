from src.motor.motor import *
import time


upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()


upDownMotor.setToInitialPosition()
time.sleep(1)
leftRightMotor.changeMode()

#Test it generates the velocity command and acc_command correctly
"""
while True:
    pos, vec = leftRightMotor.detectPositionVelocity()
    vel_cmd, acc_cmd = leftRightMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
    print("acc cmd: " + str(acc_cmd))
    time.sleep(1)
"""


while True:
    leftRightMotor.pidForConstantPosition(leftRightMotor.startPosition, Kp = 0.05, Ki = 0.05, Kd = 0)