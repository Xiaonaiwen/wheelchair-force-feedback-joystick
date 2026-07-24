from picamera2 import Picamera2
from time import perf_counter

class Camera:
    def __init__(self, width = 640, height = 480):
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
        )        
        self.camera.configure(config)
        self.camera.set_controls({
            "AeEnable": False,
            "ExposureTime": 1000,
            "AnalogueGain": 16
        })


    def start(self):
        self.camera.start()


    def stop(self):
        self.camera.stop()


    def read(self):
        frame = self.camera.capture_array("main")
        timestamp = perf_counter()
        return frame, timestamp