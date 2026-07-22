from src.motor.motor import *
import time


upDownMotor = Up_Down_Motor()
leftRightMotor = Left_Right_Motor()
graspMotor = Grasp_Motor()

upDownMotor.setToInitialPosition()
leftRightMotor.setToInitialPosition()
time.sleep(1)
# upDownMotor.changeMode()
graspMotor.changeMode()

graspMotor.runVel(445)
