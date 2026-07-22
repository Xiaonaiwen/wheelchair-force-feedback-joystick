from src.motor.motor import *
import time

upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()

leftRightMotor.setToInitialPosition()
upDownMotor.setToInitialPosition()
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

"""
while True:
    posUpDown, _ = upDownMotor.detectPositionVelocity()
    posLeftRight, _ = leftRightMotor.detectPositionVelocity()
    assignCurrent = upDownMotor.currentForGravityCompensation(posUpDown, posLeftRight)
    upDownMotor.runTorque(upDownMotor.currentBoundaryConsider(assignCurrent))
"""

while True:
    pos, vec = upDownMotor.detectPositionVelocity()
    vel_cmd, _ = upDownMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
    # Kp = 0.6, Ki = 0.0007, Kd = 0.06
    assignCurrent =  upDownMotor.pidForConstantPosition(upDownMotor.startPosition_30, Kp = 1.5, Ki = 0.7, Kd = 0.8)
    upDownMotor.runTorque(upDownMotor.currentBoundaryConsider(assignCurrent))
    

    
 