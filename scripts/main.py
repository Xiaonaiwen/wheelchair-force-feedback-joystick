from src.camera.camera import Camera
from src.vision.speedCalculate_and_terrainFrameOutput import Optical_Flow
from src.friction_model.friction_predict import predict_frame
from src.sensor.bno085 import get_acc_and_roll
from src.motor.motor import Left_Right_Motor, Up_Down_Motor, Grasp_Motor
import time

# Initialize, set parameters and start the camera
camera = Camera()
optic = Optical_Flow()
optic.default_set()
optic.set_wheel_mask(0, 220, 185, 328)
optic.set_ground_mask(0, 500, 410, 480)
camera.start()

flag = True
count = 0
previousGroundVelocity = 0
previousTime = 0
previousWheelMove = 0
previousGroundMove = 0

leftMotor = Left_Right_Motor()
graspMotor = Grasp_Motor()
upMotor = Up_Down_Motor()

# position mode to set to initial pos
upMotor.setToInitialPosition()
leftMotor.setToInitialPosition()
time.sleep(1)

# change to normal mode, use current to control the current position
upMotor.changeMode()
leftMotor.changeMode()
graspMotor.changeMode()


while flag:
    frame, currentTime = camera.read()
    success, wheel_distance_move, ground_distance_move = optic.lk(frame)
    if success:
        period = currentTime - previousTime
        slip, s, a, noMove, v, w = optic.slipRatio_and_currentAcceleration(wheel_distance_move - previousWheelMove, ground_distance_move - previousGroundMove, period, currentTime, previousGroundVelocity, previousTime)
        
        previousTime = currentTime
        previousGroundVelocity = v
        previousWheelMove = wheel_distance_move
        previousGroundMove = ground_distance_move

        print("period", period)
        print("slip: " , slip)
        print("s: ", s)
        print("noMove: ", noMove)
        print("acceleration: " , a)
        print("velocity: ", v)
        print("angular velocity: ", w)
        print()
    else:
        print("updating frame")
        previousTime = currentTime
        previousWheelMove = 0
        previousGroundMove = 0
        print()

    terrainFrame = optic.get_terrain_frame(frame)
    coefficient = predict_frame(terrainFrame)

    count += 1
    if count == 30:
        flag = False

# stop the camera
camera.stop()