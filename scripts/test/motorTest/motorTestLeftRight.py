from src.motor.motor import *
import time


upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()


upDownMotor.setToInitialPosition()
leftRightMotor.setToInitialPosition()
time.sleep(2)
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
    assignCurrent = leftRightMotor.pidForConstantPosition(leftRightMotor.startPosition, Kp = 0.46, Ki = 1, Kd = 0.08)
    leftRightMotor.runTorque(leftRightMotor.currentBoundaryConsider(assignCurrent))
    pos, vec = leftRightMotor.detectPositionVelocity()
    vel_cmd, _ = leftRightMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))