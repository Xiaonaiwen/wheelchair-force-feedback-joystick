import numpy as np
import cv2 as cv

class Optical_Flow:
    def __init__(self, width = 640, height = 480, feature_frame_refresh = 5, error_threshold = 10):
        self.width = width
        self.height = height
        self.frame_count = feature_frame_refresh
        self.refresh_bound = feature_frame_refresh
        self.error_threshold = error_threshold


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


    def set_mask(self, wheel_width_low_bound, wheel_width_high_bound, wheel_height_low_bound, wheel_height_high_bound):
        self.wheel_mask = np.zeros((self.height,self.width), dtype=np.uint8)
        self.wheel_mask[wheel_height_low_bound:wheel_height_high_bound, wheel_width_low_bound:wheel_width_high_bound] = 255
        self.ground_mask = 255 - self.wheel_mask


    def gray_frame(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return gray


    def lk(self, frame):
        current_frame = Optical_Flow.gray_frame(frame)
        if self.frame_count == self.refresh_bound:
            standard_frame = current_frame
            wheel_feature =  cv.goodFeaturesToTrack(current_frame, mask = self.wheel_mask, **self.wheel_feature_params)
            ground_feature = cv.goodFeaturesToTrack(current_frame, mask = self.ground_mask, **self.ground_feature_params)
        else:
            wheel_p, wheel_st, wheel_err = cv.calcOpticalFlowPyrLK(standard_frame, frame, wheel_feature, None, **self.wheel_lk_params)
            ground_p, ground_st, ground_err = cv.calcOpticalFlowPyrLK(standard_frame, frame, ground_feature, None, **self.wheel_lk_params)

    
