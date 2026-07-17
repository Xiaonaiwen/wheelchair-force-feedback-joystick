from src.motor.motor import *
import time

# graspMotor = Grasp_Motor()
upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()

# graspMotor.reset()
upDownMotor.setToInitialPosition()
leftRightMotor.setToInitialPosition()
time.sleep(5)
upDownMotor.changeMode()
while True:
    pos, vec = upDownMotor.detectPositionVelocity
    vel_cmd, acc_cmd = upDownMotor.transferToCmd(pos, vec)
    print("vel cmd: " + str(vel_cmd))
    print("acc cmd: " + str(acc_cmd))
    time.sleep(1)
