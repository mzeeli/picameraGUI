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
########################################
# Params
########################################
# Number of images
N = 10

# Pin BNC is connected to
pinBNC = 26

roiX = 100
roiY = 100

resX = 640
resY = 480
########################################
# Setup cameras
########################################
camera = picamera.PiCamera()

camera.color_effects = (128, 128)
camera.shutter_speed = 2000
camera.resolution = (resX, resY)
time.sleep(2)

imgs = np.empty((resY, resX, 3, N), dtype=np.uint8)

########################################
# Listen for BNC signal and save images
########################################
print("---Waiting for BNC---")
try:
    GPIO.setmode(GPIO.BCM)  # USE GPIO# as reference 
    GPIO.setup(pinBNC, GPIO.IN)

    startTime = time.time()
    while True:
        # if pin is set to high take image
        if GPIO.input(pinBNC):
            print("Detected BNC signal")
            for i in range(N):

                # Capture newly acquired images
                camera.capture(imgs[i])

                break

            timeTook = time.time() - startTime
            print(f"Finished, took {timeTook} s")
finally:
    # Upon exit reset ports to prevent damage
    GPIO.cleanup()

########################################
# Post image processing
########################################

