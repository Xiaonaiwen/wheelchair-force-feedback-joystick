from src.motor.motor import *
import time

# graspMotor = Grasp_Motor()
upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()

# graspMotor.reset()
upDownMotor.setToInitialPosition()
leftRightMotor.setToInitialPosition()
time.sleep(10)
upDownMotor.changeMode()
leftRightMotor.changeMode()
