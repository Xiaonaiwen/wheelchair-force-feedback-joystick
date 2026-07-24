import numpy as np
import cv2 as cv

class Optical_Flow:
    def __init__(self, width = 640, height = 480, feature_frame_refresh = 5, wheel_error_threshold = 10, ground_error_threshold = 10, pixel_scale = (0.57 - 0.16) / (600 - 22), slip_threshold = 0.1):
        self.width = width
        self.height = height
        
        self.frame_count = feature_frame_refresh
        self.refresh_bound = feature_frame_refresh
        self.wheel_error_threshold = wheel_error_threshold
        self.ground_error_threshold = ground_error_threshold

        #back up
        self.previous_wheel_move = 0
        self.previous_ground_move = 0    

        self.pixel_scale = pixel_scale
        self.slip_threshold = slip_threshold
        self.noMove_threshold = 1e-2




    def set_ground_feature_params(self, maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7):
        self.ground_feature_params = dict( maxCorners = maxCorners,
                                    qualityLevel = qualityLevel,
                                    minDistance = minDistance,
                                    blockSize = blockSize)


    def set_ground_lk_params(self, winSize = (15, 15), maxLevel = 2, maxIteration = 10, epsilon = 0.03):
        self.ground_lk_params = dict(  winSize  = winSize,
                                maxLevel = maxLevel,
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, maxIteration, epsilon))


    def set_wheel_feature_params(self, maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7):
        self.wheel_feature_params = dict( maxCorners = maxCorners,
                                    qualityLevel = qualityLevel,
                                    minDistance = minDistance,
                                    blockSize = blockSize)


    def set_wheel_lk_params(self, winSize = (15, 15), maxLevel = 2, maxIteration = 10, epsilon = 0.03):
        self.wheel_lk_params = dict(  winSize  = winSize,
                                maxLevel = maxLevel,
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, maxIteration, epsilon))


    def default_set(self):
        self.set_ground_feature_params()
        self.set_ground_lk_params()
        self.set_wheel_feature_params()
        self.set_wheel_lk_params()


    def set_wheel_mask(self, wheel_width_low_bound, wheel_width_high_bound, wheel_height_low_bound, wheel_height_high_bound):
        self.wheel_mask = np.zeros((self.height,self.width), dtype=np.uint8)
        self.wheel_mask[wheel_height_low_bound:wheel_height_high_bound, wheel_width_low_bound:wheel_width_high_bound] = 255

    
    def set_ground_mask(self, ground_width_low_bound, ground_width_high_bound, ground_height_low_bound, ground_height_high_bound):
        self.ground_mask = np.zeros((self.height,self.width), dtype=np.uint8)
        self.ground_mask[ground_height_low_bound:ground_height_high_bound, ground_width_low_bound:ground_width_high_bound] = 255
        self.ground_width_low_bound = ground_width_low_bound
        self.ground_width_high_bound = ground_width_high_bound
        self.ground_height_low_bound = ground_height_low_bound
        self.ground_height_high_bound = ground_height_high_bound


    def get_terrain_frame(self, frame, size = (224, 224)):
        ground = frame[self.ground_height_low_bound:self.ground_height_high_bound, self.ground_width_low_bound:self.ground_width_high_bound]
        ground = cv.resize(ground, size, interpolation = cv.INTER_LINEAR)
        return ground

    
    @staticmethod
    def gray_frame(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return gray


    def lk(self, frame):
        current_frame = Optical_Flow.gray_frame(frame)
        if self.frame_count == self.refresh_bound:
            self.standard_frame = current_frame
            self.wheel_feature =  cv.goodFeaturesToTrack(current_frame, mask = self.wheel_mask, **self.wheel_feature_params)
            self.ground_feature = cv.goodFeaturesToTrack(current_frame, mask = self.ground_mask, **self.ground_feature_params)
            
            if self.wheel_feature is None or self.ground_feature is None:
                print("frame selected does not contain features")
            else:
                self.frame_count = 0

            return False, -1, -1

        else:
            wheel_p, wheel_st, wheel_err = cv.calcOpticalFlowPyrLK(self.standard_frame, current_frame, self.wheel_feature, None, **self.wheel_lk_params)
            ground_p, ground_st, ground_err = cv.calcOpticalFlowPyrLK(self.standard_frame, current_frame, self.ground_feature, None, **self.ground_lk_params)
            
            if wheel_p is None or ground_p is None:
                print("No features discovered, use the previous one result")
                wheel_distance_move =self.previous_wheel_move
                ground_distance_move = self.previous_ground_move
            
            else:
                good_wheel_new = wheel_p[(wheel_st == 1) & (wheel_err < self.wheel_error_threshold)]
                good_wheel_reference = self.wheel_feature[(wheel_st == 1) & (wheel_err < self.wheel_error_threshold)]

                good_ground_new = ground_p[(ground_st == 1) & (ground_err < self.ground_error_threshold)]
                good_ground_reference = self.ground_feature[(ground_st == 1) & (ground_err < self.ground_error_threshold)]

                if len(good_wheel_new) == 0 or len(good_ground_new) == 0:
                    print("No good features discovered, use the previous one result")
                    wheel_distance_move = self.previous_wheel_move
                    ground_distance_move = self.previous_ground_move
                else:
                    wheel_diff = good_wheel_new.reshape(-1, 2) - good_wheel_reference.reshape(-1, 2)
                    ground_diff = good_ground_new.reshape(-1, 2) - good_ground_reference.reshape(-1, 2)

                    # Signed motion along image x-axis.
                    # Positive means movement to the right, negative means movement to the left.
                    wheel_pixel_move = np.median(wheel_diff[:, 0])
                    ground_pixel_move = np.median(ground_diff[:, 0])

                    wheel_distance_move = wheel_pixel_move * self.pixel_scale
                    ground_distance_move = ground_pixel_move * self.pixel_scale

                    self.previous_wheel_move = wheel_distance_move
                    self.previous_ground_move = ground_distance_move

            self.frame_count += 1
            return True, wheel_distance_move, ground_distance_move


    @staticmethod
    def rotational_speed(distance, time, radius):
        return distance / (time * radius)


    @staticmethod
    def ground_speed(distance, time):
        return distance / time


    def slipRatio_and_currentAcceleration(self, wheel_distance, ground_distance, period, current_time, previous_ground_velocity, previous_time, radius = 0.1778):
        w = Optical_Flow.rotational_speed(wheel_distance, period, radius)
        v = Optical_Flow.ground_speed(ground_distance, period)
        if abs(v) < self.noMove_threshold and abs(w) >= (self.noMove_threshold / radius):
            s = radius * w
            slipping = True
            noMove = False
        elif abs(v) < self.noMove_threshold and abs(w) < (self.noMove_threshold / radius):
            noMove = True
            slipping = False
            s = 0
        else:
            s = (radius * w - v) / v
            slipping = abs(s) > self.slip_threshold
            noMove = False
        a = (v - previous_ground_velocity) / (current_time - previous_time)
        return slipping, s, a, noMove, v, w