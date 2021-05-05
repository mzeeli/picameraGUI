import picamera
import RPi.GPIO as GPIO
import time
# ~ import os
import cv2
import numpy as np
import os

img = cv2.imread(r"./saved_images/BNC timing tests/bnc_test_200ms.jpg")

"""
Continuously check BNC connection and take picture when target GPIO 
pin is set to high 640, 480
"""
########################################
# Params
# 0.08, 0.09 0.1 0.1 0.1 0.11 0.11 0.12 0.13 0.14
########################################
# Number of images
N = 10

# Pin BNC is connected to
pinBNC = 26

roiX = 150
roiY = 100

resX = 640
resY = 480

# ~ img = cv2.imread(r"/home/pi/Desktop/picameraGUI/saved_images/BNC timing tests/bnc_test_2.jpg", 0)
# ~ roi = img[roiY:, roiX:roiX+200]
# ~ cv2.imshow("roi", roi)
# ~ cv2.waitKey(0)

########################################
# Setup cameras
########################################
camera = picamera.PiCamera()

camera.color_effects = (128, 128)
camera.shutter_speed = 2000
camera.framerate = 120
camera.resolution = (resX, resY)
time.sleep(2)


########################################
# Listen for BNC signal and save images
########################################
print("---Waiting for BNC---")
imgsPaths = [f"./saved_images/BNC timing tests/bnc_test_{i+1}.jpg" for i in range(N)]

try:
    GPIO.setmode(GPIO.BCM)  # USE GPIO# as reference 
    GPIO.setup(pinBNC, GPIO.IN)

    while True:
        # if pin is set to high take image
        # ~ if True:
        if GPIO.input(pinBNC):

            startTime = time.time()

            print("Detected BNC signal")

            # Capture newly acquired images
            camera.capture_sequence(imgsPaths, use_video_port=True)                

            timeTook = time.time() - startTime
            print(f"Finished, took {timeTook} s")
            break
            
finally:
    # Upon exit reset ports to prevent damage
    GPIO.cleanup()


########################################
# Post image processing
########################################
croppedImage = []
for fileName in imgsPaths:
    print(fileName)
    img = cv2.imread(fileName)
    croppedImage.append(img[roiY:, roiX:roiX+200])
            
stack1 = np.hstack((croppedImage[0], croppedImage[1], croppedImage[2], croppedImage[3], croppedImage[4]))
stack2 = np.hstack((croppedImage[5], croppedImage[6], croppedImage[7], croppedImage[8], croppedImage[9]))
fullStack = np.vstack((stack1, stack2))

cv2.imshow("Combined", fullStack)
cv2.imwrite("./saved_images/BNC timing tests/combinedImg.jpg", fullStack)
cv2.waitKey(0)
