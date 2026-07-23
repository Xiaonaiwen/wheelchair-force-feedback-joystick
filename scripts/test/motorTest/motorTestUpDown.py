from src.motor.motor import *
import time

upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()

"""
PACKETHANDLER.write1ByteTxRx(PORTHANDLER, leftRightMotor.id, TORQUE_ENABLE_ADDRESS, 0)
PACKETHANDLER.write1ByteTxRx(PORTHANDLER, leftRightMotor.id, OPERATING_MODE_ADDRESS, POSITION_CONTROL_MODE)
PACKETHANDLER.write1ByteTxRx(PORTHANDLER, leftRightMotor.id, TORQUE_ENABLE_ADDRESS, 1)
PACKETHANDLER.write4ByteTxRx(PORTHANDLER, leftRightMotor.id, GOAL_POSITION_ADDRESS, leftRightMotor.maxPosition_plus_70)
"""

leftRightMotor.setToInitialPosition()
upDownMotor.setToInitialPosition()
time.sleep(10)


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

"""
while True:
    pos, vec = upDownMotor.detectPositionVelocity()
    vel_cmd, _ = upDownMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
    assignCurrent =  upDownMotor.pidForConstantPosition(upDownMotor.startPosition_30)
    upDownMotor.runTorque(upDownMotor.currentBoundaryConsider(assignCurrent))
"""

"""
while True:
    posUpDown, vecUpDown = upDownMotor.detectPositionVelocity()
    posLeftRight, _ = leftRightMotor.detectPositionVelocity()
    assignCurrent = upDownMotor.currentForGravityCompensation(posUpDown, posLeftRight)
    vel_cmd, acc_cmd = upDownMotor.transferToCmd(posUpDown, vecUpDown)
    print("vel_cmd: " + str(vel_cmd))
    print("acc_cmd " + str(acc_cmd))
    print("pos: " + str(posUpDown))
    assignCurrent += upDownMotor.inverseTorque(10, acc_cmd, 1.2, vel_cmd, posUpDown, k_torque = 30)
    print(assignCurrent)
    print(upDownMotor.runTorque(upDownMotor.currentBoundaryConsider(assignCurrent)))
"""