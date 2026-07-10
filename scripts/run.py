from src.camera.camera import Camera
import cv2 as cv

# Initialize and start the camera
camera = Camera()
camera.start()
frame, time = camera.read()

frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

cv.imshow("Camera", frame)
cv.waitKey(0)      # Wait until a key is pressed
cv.destroyAllWindows()

camera.stop()

# stop the camera
camera.stop()