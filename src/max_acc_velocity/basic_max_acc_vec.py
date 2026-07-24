from src.friction_model.friction_predict import predict_frame
import math


class Calculator:
    def __init__(self, total_mass):
        self.g = 9.801

    def max_acceleration_velocity(self, radian, frame):
        mu = predict_frame(frame)
        N = self.g * math.cos(radian)
        maxAcc = mu * N / self.mass
        return maxAcc, maxVec