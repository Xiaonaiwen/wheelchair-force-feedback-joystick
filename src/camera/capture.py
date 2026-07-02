from picamera2 import Picamera2

class Capture:
    def __init__(self, width = 640, height = 480):
        self.camera = Picamera2()
        self.width = width
        self.height = height


    def start(self):
        config = self.camera.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
        )        
        self.camera.configure(config)

        self.camera.start()


    def stop(self):
        self.camera.stop()


    def capture_frame(self):
        return self.camera.capture_array("main")