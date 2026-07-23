from src.camera.camera import Camera
from src.vision.speedCalculate_and_terrainFrameOutput import Optical_Flow
from src.friction_model.friction_predict import predict_frame
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
previousGroundVelocity = None
previousTime = None
while flag:
    frame, currentTime = camera.read()
    success, wheel_distance_move, ground_distance_move = optic.lk(frame)
    if success:
        period = currentTime - standardFrameTime
        if previousTime is None:
            slip, s, noMove, v, w = optic.slipRatuib_and_noAcceleration(wheel_distance_move, ground_distance_move, period)
            a = None
        else:
            slip, s, a, noMove, v, w = optic.slipRatio_and_currentAcceleration(wheel_distance_move, ground_distance_move, period, currentTime, previousGroundVelocity, previousTime)
        previousTime = currentTime
        previousGroundVelocity = v
        print("period", period)
        print("slip: " , slip)
        print("s: ", s)
        print("noMove: ", noMove)
        print("acceleration: " , a)
        print("velocity: ", v)
        print("angular velocity: ", w)
        print()
    else:
        standardFrameTime = currentTime
        print("updating frame")
        print()
    terrainFrame = optic.get_terrain_frame(frame)
    coefficient = predict_frame(terrainFrame)
    count += 1
    if count == 1e3:
        flag = False

# stop the camera
camera.stop()