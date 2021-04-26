import picamera
import RPi.GPIO as GPIO
import time
# ~ import os
import cv2
import numpy as np

"""
Continuously check BNC connection and take picture when target GPIO 
pin is set to high 640, 480
"""
# Pin BNC is connected to
pinBNC = 26

# ~ img = cv2.imread("./saved_images/BNC timing tests/bnc_test.jpg")
# ~ print(np.shape(img))

saveImageName = "bnc_test.jpg"

# Setup cameras
camera = picamera.PiCamera()

camera.color_effects = (128,128)
camera.shutter_speed = 2000
camera.resolution = (640,480)
time.sleep(2)
# ~ camera.capture("test.jpg")
# ~ os.system("raspistill -t 0 -o os_test.jpg")


# Listen for BNC signal
print("---Waiting for BNC---")
try:
    GPIO.setmode(GPIO.BCM)  # USE GPIO# as reference 
    GPIO.setup(pinBNC, GPIO.IN)
    
    while True:
        # if pin is set to high take image
        if GPIO.input(pinBNC):
            
            # ~ os.syste("raspistill -o test.jpg")
            
            #Save newly acquired images 
            camera.capture(f"./saved_images/BNC timing tests/{saveImageName}")
            
            print("Detected BNC signal and saved image")
            break
                            
finally:
    # Upon exit reset ports to prevent damage
    GPIO.cleanup()
