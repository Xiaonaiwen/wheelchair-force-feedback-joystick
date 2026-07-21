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
    pos, vec = upDownMotor.detectPositionVelocity()
    vel_cmd, _ = upDownMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
    assignCurrent = upDownMotor.currentForGravityCompensation(pos)
    upDownMotor.runTorque(upDownMotor.currentBoundaryConsider(assignCurrent))
    # Kp = 0.6, Ki = 0.0007, Kd = 0.06
    assignCurrent +=  upDownMotor.pidForConstantPosition(upDownMotor.startPosition_30, Kp = 5, Ki = 1e-13, Kd = 0.1)
    upDownMotor.runTorque(upDownMotor.currentBoundaryConsider(assignCurrent))

    
    
 