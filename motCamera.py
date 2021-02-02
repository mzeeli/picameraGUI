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
    
    def __init__(self, grayscale=True):
        picamera.PiCamera.__init__(self)
        
        self.color_effects = (128,128) if grayscale else None  # By default set images to grayscale
        self.img = None  # field to store images
        

    def capImg(self, waitTime):
        self.resolution = (640, 480)
        self.start_preview()

        time.sleep(waitTime)
        self.capture('foo.jpg')
    
    def showImgOnLbl(self, label):  
        # Capture an image and display it on a tkinter label
        # label: tkinter.Label
        try:
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
        # Display continuous video stream on opencv window
        # (rx, ry): resolution to display
        
        print("Press 'q' to exit")
        self.resolution = (rx, ry)
        self.framerate = 80
         
        
        rawCapture = PiRGBArray(self)
        sleep(1)
        
        for frame in self.capture_continuous(rawCapture, format="bgr", 
                                             use_video_port=True):
            image = frame.array
        
            cv2.imshow("Press 'q' to exit", image)
            key = cv2.waitKey(1)
            rawCapture.truncate(0)
            
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
                
        
if __name__ == "__main__":
    camera = MOTCamera()
    threading.Thread(target=camera.showVid).start()

