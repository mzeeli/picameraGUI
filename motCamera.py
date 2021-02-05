from picamera.array import PiRGBArray
from PIL import ImageTk, Image
from time import sleep
from io import BytesIO

import picamera
import cv2
import tkinter as tk
import numpy as np
import threading

class MOTCamera(picamera.PiCamera):
    
    def __init__(self, camera_num, grayscale=True):
        ''' 
        camera_num [0,1]: determines which camera to open and reference
            camera_num = 0: gets camera connected to CAM1 on COMPUTE MODULE
            camera_num = 1: gets camera connected to CAM0 on COMPUTE MODULE
        '''
        
        picamera.PiCamera.__init__(self, camera_num=camera_num)
        
        # Camera Properties #
        # ~ self.sharpness = 0 # int:[-100, 100], default = 0
        # ~ self.contrast = 0   # int:[-100, 100], default = 0
        # ~ self.brightness = 50 # int:[0, 100], default = 0
        # ~ self.saturation = 0   # int:[-100, 100], default = 0
        # ~ self.iso = 0 # 100, 200, 320, 400, 500, 640, 800
        # ~ self.exposure_mode = 'fireworks' # 'night', 'nightpreview', 'backlight', 'fireworks', default = 'auto'

        # ~ self.shutter_speed  = 1800 # Shutter speed in microseconds, default = auto

        
        # Pi compute module reads cam0 port as numerical value 1 and vice versa, this converts 0->1 and 1->0
        self.camNum = abs(camera_num - 1) 
        
        self.color_effects = (128,128) if grayscale else None  # By default set images to grayscale
        self.img = None  # field to store images
        self.windowName = f"Cam{self.camNum}, double left click to exit"
        self.vidOn = False  # Tracks if video is recorded on a cv2 window

    def capImg(self, waitTime):
        self.resolution = (640, 480)
        self.start_preview()

        time.sleep(waitTime)
        self.capture('foo.jpg')
    
    def showImgOnLbl(self, label):  
        '''
        Capture an image and display it on a tkinter label
        
        label (tkinter.Label) = Target label to display snapped images
        '''
        try: # Low level error handling
            self.resolution = (608, 272)  # Default resolution to match GUI labels
            self.img = np.empty((272, 608, 3),dtype=np.uint8)

            self.capture(self.img, format="rgb", use_video_port=True)

            img = Image.fromarray(self.img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.image = imgtk
            label.configure(image=imgtk)
            
        except:
            print('Can not display image: Camera likely open in another application')


    def showVid(self, rx=640, ry=480):
        '''
        Display continuous video stream on separate opencv window, to 
        exit double left click
        
        rx (int) = x resolution
        ry (int) = y resolution
        '''
        print("Double left click to exit")
        self.resolution = (rx, ry)
        self.framerate = 80
        self.vidOn = True
        
        rawCapture = PiRGBArray(self)
        sleep(1)
        
        
        for frame in self.capture_continuous(rawCapture, format="bgr", 
                                             use_video_port=True):
            image = frame.array
            image = image[:,:,1]
            cv2.imshow(self.windowName, image)
            cv2.setMouseCallback(self.windowName, self.destroyWindows)
            
            
            key = cv2.waitKey(1)
            rawCapture.truncate(0)
            
            if not self.vidOn:
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print("Exiting video view")
                break
                
                
    def destroyWindows(self, event, x, y, flags, param):
        '''
        callback for mouse events, closes openCV window on double left click
        '''
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("Double L click detected")
            self.vidOn = False
                
        
if __name__ == "__main__":
    cam1 = MOTCamera(0)
    cam2 = MOTCamera(1)
    
    threading.Thread(target=cam1.showVid).start()



