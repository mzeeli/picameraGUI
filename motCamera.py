"""
Script for MOTCamera object - child of PiCamera class

Camera class wrapper for the more commonly known PiCamera class

The last used picameras were the NoIR V2

Last Updated: Summer Term, 2021
Author: Michael Li
"""

from picamera.array import PiRGBArray
from PIL import ImageTk, Image

import time
import picamera
import cv2
import numpy as np
import json
import io

import motAlignment


class MOTCamera(picamera.PiCamera):
    """
    The MOTCamera class is a wrapper for the PiCamera class for ease of
    access with the Pi Camera GUI. It is a child of the PiCamera class made by
    raspbian so it still maintains all it's original functionality

    Fields:
        camNum: (int [0,1]) camera number corresponding to the camera port on the
                compute module 3+. Note that camNum=1 is actually the cam0 port and
                vice versa
        cameraConfig: Decoded JSON file for previously saved camera configurations
                      such as shutter speed. Default saved to
                      ./configurations/cameraConfig.json
        color_effects: Controls grayscale
        framerate: (int) Framerate to give to the pi
        grayscale: (bool) Toggles grayscale images on the picamera
        img: (2D np.array) Last image captured with the picamera
        pixelSize: (int) Hardware constant: Physical pixel size on the picamera
        resolution: (int, int) Current resolution of the picamera
        shutter_speed: (int) Curent shutter speed of the picamera [us]
        vidOn: (bool) Tracks if video being displayed on a cv2 window
        windowName: (str) default cv2 window name for displaying video
    """
    def __init__(self, camera_num, grayscale=True):
        """
        Camera class to access camera funcitons. Child of default PiCamera class

        :param camera_num: (int [0,1]) Determines which camera to open
            camera_num = 0: gets camera connected to CAM1 on COMPUTE MODULE
            camera_num = 1: gets camera connected to CAM0 on COMPUTE MODULE
        :param grayscale: (bool) Toggles grayscale images
        """
        
        picamera.PiCamera.__init__(self, camera_num=camera_num)
        ########################################################################
        # Camera Properties
        ########################################################################
        # ~ self.sharpness = 0 # int:[-100, 100], default = 0
        # ~ self.contrast = 0   # int:[-100, 100], default = 0
        # ~ self.brightness = 50 # int:[0, 100], default = 0
        # ~ self.saturation = 0   # int:[-100, 100], default = 0
        # ~ self.iso = 0 # 100, 200, 320, 400, 500, 640, 800
        # 'night', 'nightpreview', 'backlight', 'fireworks', default = 'auto'
        # ~ self.exposure_mode = 'fireworks'

        # default = auto
        # ~ self.shutter_speed  = 800 # Shutter speed [us]

        ########################################################################
        # Initiate camera
        ########################################################################
        # Pi compute module reads cam0 port as numerical value 1 and vice versa
        # this converts 0->1 and 1->0, so camNum matches the camera port
        self.camNum = abs(camera_num - 1)
        
        self.windowName = f"Cam{self.camNum}, double left click to exit"
        self.vidOn = False  # Tracks if video is recorded on a cv2 window
        self.img = None  # field to store images

        ########################################################################
        # Configure camera
        ########################################################################
        ## Camera Hardware Constants
        self.pixelSize = 1.12e-6 ** 2  # [m^2]

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
        
    def capImgCV2(self, resX=640, resY=480):
        """
        Captures a cv2 img in grayscale and assigns it to self.img

        :param resX: (int) X resolutino of picamera
        :param resY: (int) Y resolution of picamera
        """

        if self.resolution != (resX, resY):
            self.resolution = (resX, resY)
        
        ########################################
        # Method 1: capture
        ########################################
        # ~ self.img = np.empty((resY, resX, 3), dtype=np.uint8)
        # ~ self.capture(self.img, format="bgr", use_video_port=True)
        # ~ self.img = self.img[:,:,0]
        
        ########################################
        # Method 2: capture_sequence
        # Turns out it is faster to use capture_sequence to save the image
        # Then read it rather than using capture to a variable
        ########################################
        self.capture_sequence([f"./saved_images/last_cam{self.camNum}.jpg"], use_video_port=True)
        self.img = cv2.imread(f"./saved_images/last_cam{self.camNum}.jpg", 0)
        
    def capImg2Win(self, winName="Image Capture"):
        """
        Captures still image with picameras and display them to a cv2 window

        Used to test cameras and debugging
        Note this function does not have a cv2.waitKey to show the images,
        so that has to be called outside the function where to show the 
        images. This allows multiple captures and displays to happen at 
        once.
        
        :param winName: (str) Title for cv2 display window
        """
        self.capImgCV2()
        cv2.imshow(winName, self.img)

    def capImg2File(self, fileName):
        """
        Captures a new image  and saves it to the designated filepath

        capture_sequence is much faster than capture. If you just input one filePath
        in the first argument it just captures a single image

        :param fileName: (str: filepath) Filename to save image to
        """
        self.capture_sequence(fileName, use_video_port=True)
    
    def showImgOnLbl(self, label):
        """
        Capture an image and display it on a tkinter label

        :param label: (tk.Label) Target label to display snapped images
        """
        self.capImgCV2(544, 272)
      
        roiLength = motAlignment.lengthROI
        
        if self.camNum == 1:
            roiX = motAlignment.cam1xROI
            roiY = motAlignment.cam1yROI

        else:
            roiX = motAlignment.cam0xROI
            roiY = motAlignment.cam0yROI
        imgCopy = np.copy(self.img)
        cv2.rectangle(imgCopy, (roiX, roiY), (roiX+roiLength, roiY+roiLength),
                      255, 2)

        imgCopy = Image.fromarray(imgCopy)
        imgtk = ImageTk.PhotoImage(image=imgCopy)
        label.image = imgtk
        label.configure(image=imgtk)

    def showVid(self, rx=640, ry=480):
        """
        Display continuous video stream on separate opencv window

        Note: double left click to exit
        
        :param rx: (int) x resolution
        :param ry: (int) y resolution
        """
        print("Double left click to exit")
        self.resolution = (rx, ry)
        self.framerate = 80
        self.vidOn = True
        
        rawCapture = PiRGBArray(self)
                
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
                self.resolution = (544, 272)
                self.framerate = 5
                print("Exiting video view")
                break

    def destroyWindows(self, event, x, y, flags, param):
        """
        Callback for mouse events, closes openCV window on double left click
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("Double L click detected")
            self.vidOn = False

    def calibrateShutterSpeed(self, debug=False):
        """
        Calibrates shutter speed so there are no saturated pixels

        Does so by binary search:
        The function takes an image and checks if connected pixels are 
        saturated, if they are it lowers shutter_speed and tries again.
        This process continues until an appropriate shutter_speed value
        is found.

        Todo this code was written before capImgCV2 was written and slow. Update it

        :param debug: (bool) Debugging flag
        :return: (int) calibrated shutter speed [us]
        """
        if debug:
            print("Press lower case 'q' to exit cv2 image views ")
        
        self.resolution = (544, 272)
        img = np.empty((272, 544, 3), dtype=np.uint8)
        
        # Start searching for new shutter speed from 20,000 us
        # Need starting value to be high
        startingShutter = 20000
        self.shutter_speed = 20000
        
        shutterLim = [1, startingShutter]  # shutter speed limits
        
        time.sleep(1)
        
        # Find shutter_speed value that does not lead to saturation
        foundShutter = False
        
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
                
                k = cv2.waitKey(0)
                if k == ord("q"):
                    break
                    
            else:
                # Even if not debugging still need to delay camera images
                #to give camera time warm up and change shutter speeds
                time.sleep(1)

        if debug:
            cv2.destroyAllWindows()

        print(f"new shutter speed: {self.shutter_speed}")
        self.cameraConfig['shutter_speed'] = self.shutter_speed
        with open(r"./configurations/cameraConfig.json", "w") as outFile:
            json.dump(self.cameraConfig, outFile)
            
        return self.shutter_speed

if __name__ == "__main__":
    #################################################
    # calibration tests
    #################################################
    # ~ cam1 = MOTCamera(0)
    # ~ cam1.capImg2Win(f"Before Calibration, shutter speed = {cam1.shutter_speed}")
    
    # ~ cam1.showVid()
    
    # ~ cam1.calibrateShutterSpeed(debug=True)
    
    
    # ~ cam1.capImg2Win(f"After Calibration0, shutter speed = {cam1.shutter_speed}")
    # ~ time.sleep(1)
    # ~ cam1.capImg2Win(f"After Calibration1, shutter speed = {cam1.shutter_speed}")
    # ~ cv2.waitKey(0)
    
    #################################################
    # io stream tests
    #################################################
    cam1 = MOTCamera(0)
    stream = io.BytesIO()
    
    # ~ cam1.start_preview()
    time.time.sleep(2)
    cam1.capture(stream, format="jpeg")
    
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)
    print(np.shape(img))
    cv2.imshow("img", img)
    cv2.waitKey(0)
