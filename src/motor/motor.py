#!/usr/bin/env python3
from dynamixel_sdk import *
import time
import math

pi = "/dev/ttyUSB0"
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

#Wheelchair Property
RPM_TO_SPEEDUNIT = 320 / 73.24
WHEELCHAIR_MAX_SPEED_METER_PER_SECOND = 3
WHEELCHAIR_MAX_ACCELERATION_METER_PER_SQUARE_SECOND = 1.2
SAFETY_PERIOD = 2
WHEELCHAIR_MAX_CURRENT = 15


class Left_Right_Motor():
    def __init__(self):
        self.startPosition = 2048
        self.minPosition_minus_70 = 1252
        self.maxPosition_plus_70 = 2844
        self.max_speed_unit = WHEELCHAIR_MAX_ACCELERATION_METER_PER_SQUARE_SECOND / WHEELCHAIR_MAX_SPEED_METER_PER_SECOND * (self.maxPosition_plus_70 - self.startPosition) / 4096 * 60 * RPM_TO_SPEEDUNIT
        self.currentLimit = 910

        self.integral = 0
        self.previousError = None
        self.previousTime = None

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

    def pidReset(self):
        self.integral = 0
        self.previousError = None
        self.previousTime = None

    def pidForConstantPosition(self, goalPos, Kp = 0.2, Ki = 0.5, Kd = 0.04):
        currentTime = time.perf_counter()
        currentPos, _ = self.detectPositionVelocity()
        currentError = goalPos - currentPos
        
        if self.previousTime is None:
            self.previousTime = currentTime
            self.previousError = currentError
            currentOutput = Kp * currentError
        else:
            timePeriod = currentTime - self.previousTime
            self.integral += currentError * timePeriod
            if Ki == 0:
                integralLimit = float('inf')
            else:
                integralLimit = self.currentLimit / Ki / 1.4
            self.integral = max(min(self.integral, integralLimit), -integralLimit)
            currentOutput = currentError * Kp + (currentError - self.previousError) / timePeriod * Kd + self.integral * Ki
            self.previousTime = currentTime
            self.previousError = currentError
        return currentOutput

    def currentBoundaryConsider(self, currentEnter):
        currentOutput = int(max(min(currentEnter, self.currentLimit), -self.currentLimit))
        return currentOutput

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
        PACKETHANDLER.write2ByteTxRx(PORTHANDLER, self.id, GOAL_CURRENT_ADDRESS, current & 0xFFFF)


    def inverseTorque(
        self, 

        wheel_current, 
        acc_cmd, 
        acc_max, 
        vel_cmd, 
        pos, 

        k_torque = 0, 
        k_acc = 0, 
        k_vel = 0, 
        k_boundary = 0, 
        k_initial_pos = 0, 
        boundary_range = 10, 
        initial_range = 10
        ):

        vel_max = acc_max * SAFETY_PERIOD
        # transfer to command unit
        acc_max = acc_max / WHEELCHAIR_MAX_ACCELERATION_METER_PER_SQUARE_SECOND * 100
        vel_max = vel_max / WHEELCHAIR_MAX_SPEED_METER_PER_SECOND * 100
        acc_diff = max(abs(acc_cmd) - acc_max, 0)
        vel_diff = max(abs(vel_cmd) - vel_max, 0)

        if acc_cmd < 0:
            acc_diff *= -1
        if vel_cmd < 0:
            vel_diff *= -1
        
        if pos < startPosition:
            boundary_diff = min(pos - self.minPosition_minus_70 - boundary_range, 0)
        else:
            boundary_diff = max(pos - self.maxPosition_plus_70 + boundary_range, 0)
        
        initial_diff = 0
        if pos < (self.startPosition + initial_range) and pos > (self.startPosition - initial_range):
            initial_diff = abs(pos - self.startPosition)
            if pos < self.startPosition:
                initial_diff *= -1

        current = k_torque * wheel_current + k_acc * acc_diff + k_vel * vel_diff + k_boundary * boundary_diff + k_initial_pos * initial_diff
        return current


class Up_Down_Motor():
    def __init__(self):
        self.startPosition_30 = 2389
        self.minPosition_0 = 2048
        self.maxPosition_60 = 2731
        self.max_speed_unit = WHEELCHAIR_MAX_ACCELERATION_METER_PER_SQUARE_SECOND / WHEELCHAIR_MAX_SPEED_METER_PER_SECOND * (self.maxPosition_60 - self.startPosition_30) / 4096 * 60 * RPM_TO_SPEEDUNIT
        self.currentLimit = 910

        self.integral = 0
        self.previousError = None
        self.previousTime = None

        self.gravityCompensationHorizontal = 160

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
    
    def pidReset(self):
        self.integral = 0
        self.previousError = None
        self.previousTime = None


    def pidForConstantPosition(self, goalPos, Kp = 1.1, Ki = 1.4, Kd = 0.2):
        currentTime = time.perf_counter()
        currentPos, _ = self.detectPositionVelocity()
        currentError = goalPos - currentPos
        if self.previousTime is None:
            self.previousTime = currentTime
            self.previousError = currentError
            currentOutput = Kp * currentError
        else:
            timePeriod = currentTime - self.previousTime
            self.integral += currentError * timePeriod
            if Ki == 0:
                integralLimit = float('inf')
            else:
                integralLimit = self.currentLimit / Ki / 1.4
            self.integral = max(min(self.integral, integralLimit), -integralLimit)
            currentOutput = currentError * Kp + (currentError - self.previousError) / timePeriod * Kd + self.integral * Ki
            self.previousTime = currentTime
            self.previousError = currentError
        return currentOutput

    def currentForGravityCompensation(self, posUpDown, posLeftRight):
        angleLeftRight_radian = abs(posLeftRight - 2048) * 2 * math.pi / 4096
        angleFromHorizontal_radian = abs(posUpDown - self.minPosition_0) * 2 * math.pi / 4096
        currentCompensation = self.gravityCompensationHorizontal * math.cos(angleFromHorizontal_radian) * math.cos(angleLeftRight_radian)
        if (posLeftRight - 2048) > 0:
            currentCompensation *= 1.3
        return currentCompensation

    def currentBoundaryConsider(self, currentEnter):
        currentOutput = int(max(min(currentEnter, self.currentLimit), -self.currentLimit))
        return currentOutput

    def transferToCmd(self, pos, vec):
        # pos to vel_cmd, vec to acc_cmd all in percentage
        if pos < self.startPosition_30:
            vel_cmd_percentage = (self.startPosition_30 - pos) / (self.startPosition_30 - self.minPosition_0) * 100
            vel_cmd = min(int(vel_cmd_percentage), 100) 
        else:
            vel_cmd_percentage = (pos - self.startPosition_30) / (self.maxPosition_60 - self.startPosition_30) * 100
            vel_cmd = min(int(vel_cmd_percentage), 100) * -1

        speed = abs(vec)
        abs_acc_cmd_percentage = min(int(speed / self.max_speed_unit * 100), 100)
        if vec < 0:
            acc_cmd = abs_acc_cmd_percentage 
        else:
            acc_cmd = abs_acc_cmd_percentage * -1

        return vel_cmd, acc_cmd
 
    def runTorque(self, current):
        dxl_comm_result, _ = PACKETHANDLER.write2ByteTxRx(PORTHANDLER, self.id, GOAL_CURRENT_ADDRESS, current & 0xFFFF)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication failed: {PACKETHANDLER.getTxRxResult(dxl_comm_result)}")
            return False
        else:
            return True


    def inverseTorque(
        self, 

        wheel_current, 
        acc_cmd, 
        acc_max, 
        vel_cmd, 
        pos, 

        k_torque = 0, 
        k_acc = 0, 
        k_vel = 0, 
        k_boundary = 0, 
        k_initial_pos = 0, 
        boundary_range = 10, 
        initial_range = 10
        ):

        vel_max = acc_max * SAFETY_PERIOD
        # transfer to command unit
        acc_max = acc_max / WHEELCHAIR_MAX_ACCELERATION_METER_PER_SQUARE_SECOND * 100
        vel_max = vel_max / WHEELCHAIR_MAX_SPEED_METER_PER_SECOND * 100

        acc_diff = max(abs(acc_cmd) - acc_max, 0)
        vel_diff = max(abs(vel_cmd) - vel_max, 0)
        if acc_cmd < 0:
            acc_diff *= -1
        if vel_cmd < 0:
            vel_diff *= -1
        
        if pos < self.startPosition_30:
            boundary_diff = min(pos - self.minPosition_0 - boundary_range, 0)
        else:
            boundary_diff = max(pos - self.maxPosition_60 + boundary_range, 0)
        
        initial_diff = 0
        if pos < (self.startPosition_30 + initial_range) and pos > (self.startPosition_30 - initial_range):
            initial_diff = abs(pos - self.startPosition_30)
            if pos < self.startPosition_30:
                initial_diff *= -1

        current = k_torque * wheel_current + k_acc * acc_diff + k_vel * vel_diff + k_boundary * boundary_diff + k_initial_pos * initial_diff
        return current

class Grasp_Motor():
    def __init__(self):
        self.max_speed_unit = 445
        self.max_speed_rpm = 101.85
        self.id = GRASP_IDX

    def changeMode(self):
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 0)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, OPERATING_MODE_ADDRESS, VELOCITY_CONTROL_MODE)
        PACKETHANDLER.write1ByteTxRx(PORTHANDLER, self.id, TORQUE_ENABLE_ADDRESS, 1)

    def runVel(self, vel):
        PACKETHANDLER.write4ByteTxRx(PORTHANDLER, self.id, GOAL_VELOCITY_ADDRESS, vel & 0xFFFFFFFF)
       
    def reverseSpeed(self, forwardFlag, slip_ratio, maximumSlipExpected = 5, maxSpeed = 445):
        if slip_ratio < 0.1:
            slip_ratio = 0

        k_slip = maxSpeed / maximumSlipExpected
        speed = slip_ratio * k_slip
        speed = int(max(min(speed, maxSpeed), -maxSpeed))

        if forwardFlag:
            vel = speed * -1
            self.runVel(vel) 
        else:
            vel = speed
            self.runVel(vel)






