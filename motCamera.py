"""
Script for MOTCamera object - child of PiCamera class

Last Updated: Winter, 2021
Author: Michael Li
"""

from picamera.array import PiRGBArray
from PIL import ImageTk, Image
from time import sleep

import picamera
import cv2
import numpy as np
import threading
import json

class MOTCamera(picamera.PiCamera):
    
    def __init__(self, camera_num, grayscale=True):
        """
        camera class to access camera funcitons. Child of default PiCamera class

        :param camera_num: (int [0,1]) determines which camera to open and
        reference
            camera_num = 0: gets camera connected to CAM1 on COMPUTE MODULE
            camera_num = 1: gets camera connected to CAM0 on COMPUTE MODULE
        :param grayscale:
        """
        
        picamera.PiCamera.__init__(self, camera_num=camera_num)
        
        # Camera Properties #
        # ~ self.sharpness = 0 # int:[-100, 100], default = 0
        # ~ self.contrast = 0   # int:[-100, 100], default = 0
        # ~ self.brightness = 50 # int:[0, 100], default = 0
        # ~ self.saturation = 0   # int:[-100, 100], default = 0
        # ~ self.iso = 0 # 100, 200, 320, 400, 500, 640, 800
        # 'night', 'nightpreview', 'backlight', 'fireworks', default = 'auto'
        # ~ self.exposure_mode = 'fireworks'

        # default = auto
        # ~ self.shutter_speed  = 800 # Shutter speed [us]

        # Pi compute module reads cam0 port as numerical value 1 and vice versa
        # this converts 0->1 and 1->0, so camNum matches the camera port
        self.camNum = abs(camera_num - 1)
        
        self.windowName = f"Cam{self.camNum}, double left click to exit"
        self.vidOn = False  # Tracks if video is recorded on a cv2 window
        self.img = None  # field to store images
        
        ## Camera Hardware Constants
        self.pixelSize = 1.12e-6 ** 2

        ## Image dimension settings
        # By default set images to grayscale
        self.grayscale = grayscale
        self.color_effects = (128, 128) if grayscale else None

        ## Image capture settings
        # Load configurations from config file
        configFile = open(r"./configurations/cameraConfig.json", "r")
        cameraConfig = configFile.read()
        self.cameraConfig = json.loads(cameraConfig)
                
        self.resolution = (640, 480)
        self.framerate = int(self.cameraConfig["framerate"])
        self.shutter_speed = int(self.cameraConfig["shutter_speed"])


    def capImg2Win(self, winName="Image Capture", waitTime=1):
        """
        Captures still image with picameras and display them to a cv2 window
        Used to test cameras and debugging

        Note this function does not have a cv2.waitKey to show the images,
        so that has to be called outside the function where to show the 
        images. This allows multiple captures and displays to happen at 
        once.
        
        :param winName: (str) Title for cv2 display window
        :param waitTime: Time delay for camera warmup and capture img [s]
        
        :return:
        """
        self.resolution = (640, 480)
        img = np.empty((480, 640, 3), dtype=np.uint8)
        self.capture(img, format="bgr", use_video_port=True)

        print(f"255 exist: {255 in img}")

        cv2.imshow(winName, img)
        
        # ~ self.start_preview()
        # ~ sleep(waitTime)
        # ~ self.capture('foo.jpg')
    
    def showImgOnLbl(self, label):
        """
        Capture an image and display it on a tkinter label

        label (tkinter.Label) = Target label to display snapped images
        """
        self.resolution = (608, 272)  # Default resolution to match GUI labels
        self.img = np.empty((272, 608, 3), dtype=np.uint8)

        self.capture(self.img, format="rgb", use_video_port=True)
        img = Image.fromarray(self.img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.image = imgtk
        label.configure(image=imgtk)



    def showVid(self, rx=640, ry=480):
        """
        Display continuous video stream on separate opencv window, to 
        exit double left click
        
        rx (int) = x resolution
        ry (int) = y resolution
        """
        print("Double left click to exit")
        self.resolution = (rx, ry)
        self.framerate = 80
        self.vidOn = True
        
        rawCapture = PiRGBArray(self)
        sleep(1)
                
        for frame in self.capture_continuous(rawCapture, format="bgr", 
                                             use_video_port=True):
            image = frame.array      
            
            # Take first dimension as grayscale
            if self.grayscale:
                image = image[:, :, 1]
            
            
            cv2.imshow(self.windowName, image)
            
            # Create call back for double left click closes window
            cv2.setMouseCallback(self.windowName, self.destroyWindows)

            cv2.waitKey(1)
            rawCapture.truncate(0)
            
            if not self.vidOn:
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print("Exiting video view")
                break
                
                
    def destroyWindows(self, event, x, y, flags, param):
        """
        callback for mouse events, closes openCV window on double left click
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("Double L click detected")
            self.vidOn = False


    def calibrateShutterSpeed(self, debug=False):
        """
        Finds self.shutter_speed value that does not lead to consecutive 
        saturated pixels in an image. Does so by binary search
        
        The function takes an image and checks if connected pixels are 
        saturated, if they are it lowers shutter_speed and tries again.
        This process continues until an appropriate shutter_speed value
        is found.
        
        
        """
        if debug:
            print("Press lower case 'q' to exit cv2 image views ")
        
        self.resolution = (608, 272)
        img = np.empty((272, 608, 3), dtype=np.uint8)
        
        # Take an image only to warm up the camera, data not used
        self.capture(img, format="bgr", use_video_port=True)
        
        # Start searching for new shutter speed from 200,000 us
        # Need starting value to be high
        startingShutter = 200000
        self.shutter_speed = 200000
        shutterLim = [0, startingShutter]  # shutter speed limits
        
        # Find shutter_speed value that does not lead to saturation
        foundShutter = False
        
        # Add tolerance for noise

        while not foundShutter:
                        
            # Snap image
            self.capture(img, format="bgr", use_video_port=True)

            # Mask for intensities > 253
            ret, mask = cv2.threshold(img[:, :, 1], 253, 255, 
                                      cv2.THRESH_BINARY)
            
            # If saturated value in the mask decrase shutter speed
            
            if 255 in mask:
                shutterLim[1] = self.shutter_speed
                
            # Use first unsaturated shutter speed
            else:
                shutterLim[0] = self.shutter_speed
            
            prevShutterSpeed = self.shutter_speed
            self.shutter_speed = int((shutterLim[0] + shutterLim[1]) / 2)

            
            # If speed stays the same for two iterations exit search
            if prevShutterSpeed == self.shutter_speed:
                if debug:
                    print(f"New calibrated shutter speed: {self.shutter_speed}")

                foundShutter = True
            
            if debug:
                print(f"shutterLim: {shutterLim}")
                print(f"next shutter_speed: {self.shutter_speed}")
                print(30*"-")
                
                cv2.imshow("mask", mask)
                cv2.imshow("img", img)
                
                k = cv2.waitKey(400)
                if k == ord("q"):
                    break
                    
            else:
                # Even if not debugging still need to delay camera images
                #for consistency
                sleep(1)

        print(f"new shutter speed: {self.shutter_speed}")
        self.cameraConfig['shutter_speed'] = self.shutter_speed
        with open(r"./configurations/cameraConfig.json", "w") as outFile:
            json.dump(self.cameraConfig, outFile)
            
        return self.shutter_speed
        
if __name__ == "__main__":
    cam1 = MOTCamera(0)
    cam1.capImg2Win(f"Before Calibration, shutter speed = {cam1.shutter_speed}")
    
    # ~ cam1.showVid()
    
    cam1.calibrateShutterSpeed(debug=True)
    
    
    cam1.capImg2Win(f"After Calibration0, shutter speed = {cam1.shutter_speed}")
    sleep(1)
    cam1.capImg2Win(f"After Calibration1, shutter speed = {cam1.shutter_speed}")
    cv2.waitKey(0)
    
    # ~ threading.Thread(target=cam1.showVid).start()
