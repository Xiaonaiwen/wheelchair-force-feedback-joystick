#!/usr/bin/env python3
from dynamixel_sdk import *
import time

pi = "/dev/ttyUSB0"
window_device = "COM7"
PORTHANDLER = PortHandler(pi)
PACKETHANDLER = PacketHandler(2.0)

PORTHANDLER.openPort()
PORTHANDLER.setBaudRate(57600)

# Mode the motor use
OPERATING_MODE_ADDRESS = 11
CURRENT_CONTROL_MODE = 0
VELOCITY_CONTROL_MODE = 1
POSITION_CONTROL_MODE = 3

# things the motor need to changes
TORQUE_ENABLE_ADDRESS = 64
GOAL_CURRENT_ADDRESS = 102
GOAL_VELOCITY_ADDRESS = 104
GOAL_POSITION_ADDRESS = 116
PRESENT_POSITION_ADDRESS = 132
PRESENT_VELOCITY_ADDRESS = 128


# motor idx 
LEFT_RIGHT_IDX = 1
UP_DOWN_IDX = 2
GRASP_IDX = 3


class Left_Right_Motor():
    def __init__(self, k_torque = 0, k_acc = 0, k_vel = 0):
        self.k_torque = k_torque
        self.k_acc = k_acc
        self.k_vel = k_vel
        self.startPosition = 2048
        self.minPosition_minus_70 = 1252
        self.maxPosition_plus_70 = 2844
        self.max_speed_unit = 320
        self.currentLimit = 910
        self.id = LEFT_RIGHT_IDX

    def setToInitialPosition(self):
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 0)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, OPERATING_MODE_ADDRESS, POSITION_CONTROL_MODE)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 1)
        PACKETHANDLER.write4ByteTxRx(PORTHANDLER, self.id, GOAL_POSITION_ADDRESS, self.startPosition)

    def changeMode(self):
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 0)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, OPERATING_MODE_ADDRESS, CURRENT_CONTROL_MODE)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 1)

    def detectPositionVelocity(self):
        pos, _, _ = PACKETHANDLER.read4ByteTxRx(PORTHANDLER, self.id, PRESENT_POSITION_ADDRESS)
        vec, _, _ = PACKETHANDLER.read4ByteTxRx(PORTHANDLER, self.id, PRESENT_VELOCITY_ADDRESS)
        if vec > 0x7FFFFFFF:
            vec -= 0x100000000
        return pos, vec

    def transferToCmd(self, pos, vec):
        # pos to vel_cmd, vec to acc_cmd all in percentage
        if pos < self.startPosition:
            vel_cmd_percentage = (self.startPosition - pos) / (self.startPosition - self.minPosition_minus_70) * 100
            vel_cmd = min(int(vel_cmd_percentage), 100) * -1
        else:
            vel_cmd_percentage = (pos - self.startPosition) / (self.maxPosition_plus_70 - self.startPosition) * 100
            vel_cmd = min(int(vel_cmd_percentage), 100) 

        speed = abs(vec)
        abs_acc_cmd_percentage = min(int(speed / self.max_speed_unit * 100), 100)
        if vec < 0:
            acc_cmd = abs_acc_cmd_percentage * -1
        else:
            acc_cmd = abs_acc_cmd_percentage

        return vel_cmd, acc_cmd
      
    

    def runTorque(self, current):
        print("current limit: " + str(self.currentLimit))
        PACKETHANDLER.write2ByteTxRx(PORTHANDLER, self.id, GOAL_CURRENT_ADDRESS, current & 0xFFFF)

    def inverseTorque(self, wheel_current, acc_cmd, acc_max, vel_cmd, vel_max):
        current = int(self.k_torque * wheel_current + self.k_acc * max(acc_cmd - acc_max, 0) + self.k_vel * max(vel_cmd - vel_max, 0))
        self.runTorque(current)


class Up_Down_Motor():
    def __init__(self, k_torque = 0, k_acc = 0, k_vel = 0):
        self.k_torque = k_torque
        self.k_acc = k_acc
        self.k_vel = k_vel
        self.startPosition_30 = 2389
        self.minPosition_0 = 2048
        self.maxPosition_60 = 2731
        self.max_speed_unit = 320
        self.currentLimit = 910
        self.currentLimit = 910
        self.id = UP_DOWN_IDX

    def setToInitialPosition(self):
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 0)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, OPERATING_MODE_ADDRESS, POSITION_CONTROL_MODE)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 1)
        PACKETHANDLER.write4ByteTxRx(PORTHANDLER, self.id, GOAL_POSITION_ADDRESS, self.startPosition_30)

    def changeMode(self):
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 0)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, OPERATING_MODE_ADDRESS, CURRENT_CONTROL_MODE)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 1)

    def detectPositionVelocity(self):
        pos, _, _ = PACKETHANDLER.read4ByteTxRx(PORTHANDLER, self.id, PRESENT_POSITION_ADDRESS)
        vec, _, _ = PACKETHANDLER.read4ByteTxRx(PORTHANDLER, self.id, PRESENT_VELOCITY_ADDRESS)
        if vec > 0x7FFFFFFF:
            vec -= 0x100000000
        return pos, vec
    
    def transferToCmd(self, pos, vec):
        # pos to vel_cmd, vec to acc_cmd all in percentage
        if pos < self.startPosition_30:
            vel_cmd_percentage = (self.startPosition_30 - pos) / (self.startPosition_30 - self.minPosition_0) * 100
            vel_cmd = min(int(vel_cmd_percentage), 100) * -1
        else:
            vel_cmd_percentage = (pos - self.startPosition_30) / (self.maxPosition_60 - self.startPosition_30) * 100
            vel_cmd = min(int(vel_cmd_percentage), 100) 

        speed = abs(vec)
        abs_acc_cmd_percentage = min(int(speed / self.max_speed_unit * 100), 100)
        if vec < 0:
            acc_cmd = abs_acc_cmd_percentage * -1
        else:
            acc_cmd = abs_acc_cmd_percentage

        return vel_cmd, acc_cmd
 
    def runTorque(self, current):
        print("current limit: " + str(self.currentLimit))
        PACKETHANDLER.write2ByteTxRx(PORTHANDLER, self.id, GOAL_CURRENT_ADDRESS, current & 0xFFFF)

    def inverseTorque(self, wheel_current, acc_cmd, acc_max, vel_cmd, vel_max):
        current = int(self.k_torque * wheel_current + self.k_acc * max(acc_cmd - acc_max, 0) + self.k_vel * max(vel_cmd - vel_max, 0))
        PACKETHANDLER.write2ByteTxRx(PORTHANDLER, self.id, GOAL_CURRENT_ADDRESS, current & 0xFFFF)
        time.sleep(0.5)
        PACKETHANDLER.write2ByteTxRx(PORTHANDLER, self.id, GOAL_CURRENT_ADDRESS, 0)


class Grasp_Motor():
    def __init__(self, k_slip):
        self.k_slip = k_slip
        self.max_speed_unit = 445
        self.max_speed_rpm = 101.85
        self.id = GRASP_IDX

    def changeMode(self):
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 0)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, OPERATING_MODE_ADDRESS, VELOCITY_CONTROL_MODE)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 1)

    def runVel(self, vel):
        print("max speed unit: " + str(self.max_speed_unit))
        PACKETHANDLER.write4ByteTxRx(PORTHANDLER, self.id, GOAL_VELOCITY_ADDRESS, vel & 0xFFFFFFFF)
       
    def speed(self, rotation_speed, slip_ratio):
        if slip_ratio < 0:
            slip_ratio = 0
 
        speed = slip_ratio * self.k_slip
        if rotation_speed >= 0:
            vel = speed
            self.runVel(vel)
        else:
            vel = speed * -1
            self.runVel(vel)






