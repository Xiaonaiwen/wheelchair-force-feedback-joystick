from src.friction_model.friction_predict import predict_frame
import math


class Calculator:
    def __init__(self, total_mass, safety_period = 2):
        self.mass = total_mass
        self.safety_period = safety_period
        self.g = 9.801

    def max_acceleration_velocity(self, radian, frame):
        mu = predict_frame(frame)
        N = self.m * self.g * math.cos(radian)
        maxAcc = mu * N / self.mass
        maxVec = maxAcc * self.safety_period
        return maxAcc, maxVec