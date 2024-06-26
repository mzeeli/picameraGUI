"""
Wrapper for pyueye IDS camera functions. Use the Thorcam class to connect 
and adjust the thorcam camera parameters.

The thorcams last used were the - model. The rpi compute
module 3+ has the capability to connect to two thorcams. It's important
to make sure the usb hub is capable of handling the cameras.

Last installed drivers were 4.92. Technically meant for the raspberry pi 3
but they work on the compute module 3+ as well.

Last Updated: Summer Term, 2021
Author: Michael Li
"""

from pyueye import ueye
import numpy as np
import cv2

import time


class Thorcam():
    """
    The Thorcam class is used to connect and take images with a thorcam.

    This class is a wrapper to make working with pyueye easier. Some of the
    original function names in pyueye are not straight forward and difficult to
    work with. This class hopes to streamline the whole process

    Fields:
        hCam: (ueye.HIDS) Camera handle
        height: (int) Height of camera AOI resolution
        lineinc: (float) Line increment
        mem_id: (ueye.int) Developer ID for camera
        mem_ptr: (ueye.c_mem_p) Pointer allocated to camera image memory
        nBitsPerPixel: (int) Bits per pixel
        width: (int) Width of camera AOI resolution

    """
    def __init__(self, camID, width=1280, height=1080, trigTimeout=0):
        """
        Constructor

        :param camID: (int [1,2]) determines which camera to open
        :param width: (int) width of image
        :param height: (int) height of image
        :param trigTimeout: (int) Trigger mode timeout in 10ms.
                            i.e. trigTimeout = 10 results in 100ms timeout.
                            0 will set it to default value of 60 seconds
        """

        # init camera
        self.hCam = ueye.HIDS(camID)
        ret = ueye.is_InitCamera(self.hCam, None)

        #################################################################
        # Camera Parameters
        #################################################################
        # set color mode
        ret = ueye.is_SetColorMode(self.hCam, ueye.IS_CM_BGR8_PACKED)

        # set region of interest
        self.width = width
        self.height = height
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(0)
        rect_aoi.s32Y = ueye.int(0)
        rect_aoi.s32Width = ueye.int(self.width)
        rect_aoi.s32Height = ueye.int(self.height)
        ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi,
                    ueye.sizeof(rect_aoi))

        # set timeout value for trigger mode
        ret = ueye.is_SetTimeout(self.hCam, ueye.IS_TRIGGER_TIMEOUT, trigTimeout)

        #################################################################
        # Memory
        #################################################################
        # allocate memory
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()
        self.nBitsPerPixel = 24  # for colormode = IS_CM_BGR8_PACKED
        ret = ueye.is_AllocImageMem(self.hCam, self.width, self.height,
                                    self.nBitsPerPixel, self.mem_ptr, self.mem_id)

        # set active memory region
        ret = ueye.is_SetImageMem(self.hCam, self.mem_ptr, self.mem_id)
        self.lineinc = self.width * int((self.nBitsPerPixel + 7) / 8)

    def getExposureTime(self):
        """
        retrieves camera exposure time
		
        :return: (ueye.double) Exposure time in ms
        """
        exposureTime = ueye.double()
        nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, time_output, 8)
        return exposureTime

    def setExposureTime(self, newExpTime, debug=False, dumpImage=True):
        """
        Sets a new exposure time for the thorcam

        Note: For the new exposure time to be set, need to first take image without
        doing anything. The first image from camera is always set to the previous
        exposure time, so a work around is to throw away the first image everytime
        when setting a new one

        :param newExpTime: (float) New exposure time [ms]
        :param debug: (bool) debugging flag
        :param dumpImage: (bool) whether or not to throw out first image
        """
        # Set exposure time
        time_input = ueye.double(newExpTime)
        nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                                time_input, 8)
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Input ERROR")

        # Instantly take image and dump it, see docstring
        if dumpImage:
            self.capImgNow()

        if debug:
            # get exposure time range
            time_min = ueye.double(0)
            nRet = ueye.is_Exposure(self.hCam,
                                    ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN,
                                    time_min, 8)
            if nRet != ueye.IS_SUCCESS:
                print("is_Exposure Min. Range ERROR")
            print("Get Exposure Time Min.", time_min)

            time_max = ueye.double(0)
            nRet = ueye.is_Exposure(self.hCam,
                                    ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX,
                                    time_max, 8)
            if nRet != ueye.IS_SUCCESS:
                print("is_Exposure Max. Range ERROR")
            print("Get Exposure Time Max.", time_max)

            # Verify newly set exposure time
            time_output = ueye.double()
            nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                                    time_output, 8)
            if nRet != ueye.IS_SUCCESS:
                print("is_Exposure Output ERROR")
            print("Get Exposure Time", time_output)

            # Check default exposure time
            time_default = ueye.double()
            nRet = ueye.is_Exposure(self.hCam,
                                    ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_DEFAULT,
                                    time_default, 8)
            if nRet != ueye.IS_SUCCESS:
                print("is_InitCamera ERROR")

            print("Get Default Exposure Time", time_default)

    def capImgNow(self) -> np.array:
        """
        Instantly captures an image with software trigger
        :return: (3D numpy.array) captured image
        """
        # 0: instant image, 2: set trigger
        ret = ueye.is_SetExternalTrigger(self.hCam, 0)

        # Wait until 1st image is in memory, timing depends on is_SetExternalTrigger
        ret = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
        img = ueye.get_data(self.mem_ptr, self.width, self.height,
                            self.nBitsPerPixel, self.lineinc, copy=True)

        img = np.reshape(img, (self.height, self.width, 3))
        return img

    def showVid(self):		
        while True:
            # Capture image via hardware trigger
            img = self.capImgNow()

            cv2.imshow("Thorcam, press 'q' to exit absorption imaging", img)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                print("Exit capture loop")
                break
        cv2.destroyAllWindows()
            
    def enableHardwareTrig(self) -> np.array:
        """
        Starts hardware trigger for thorcam. Program waits on line
        ueye.is_FreezeVideo until a hardware trigger captures an image with the
        camera

        :return: (3D numpy.array) captured image
        """
        # 0: instant image, 2: set trigger (low -> high)
        ret = ueye.is_SetExternalTrigger(self.hCam, 2)

        # Wait until 1st image is in memory, timing depends on is_SetExternalTrigger
        ret = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
        img = ueye.get_data(self.mem_ptr, self.width, self.height,
                            self.nBitsPerPixel, self.lineinc, copy=True)

        img = np.reshape(img, (self.height, self.width, 3))
        return img


    def startAbsorptionImaging(self):
		# Todo confirm with Paul what timings we should look for 
		# 100 here is extermely high, don't expect to get > 30
        for i in range(2, 100, 2):
            print(f"----{i}ms----")
            print("Waiting")

            # Capture image via hardware trigger
            img = self.enableHardwareTrig()

            cv2.imshow(f"{i}ms, press 'q' to exit absorption imaging", img)
            cv2.imwrite(f"./images/{i}ms.jpg", img)

            key = cv2.waitKey(0)
            if key == ord('q'):
                print("Exit capture loop")
                break

            cv2.destroyAllWindows()

    def capVideo():
        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(hCam, ueye.IS_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press q to leave the programm")

        #---------------------------------------------------------------------------------------------------------------------------------------
        # Single Image Capture
        while(nRet == ueye.IS_SUCCESS):            
            # In order to display the image in an OpenCV window we need to...
            # ...extract the data of our image memory
            array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

            bytes_per_pixel = int(nBitsPerPixel / 8)
            
            # ...reshape it in an numpy array...
            frame = np.reshape(array,(height.value, width.value, bytes_per_pixel))
            # # ...resize the image by a half
            frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                        #---------------------------------------------------------------------------------------------------------------------------------------
                #Include image data processing here
                #---------------------------------------------------------------------------------------------------------------------------------------
                #     #...and finally display it
            
            cv2.imshow("Thorcam", frame)
            key = cv2.waitKey(0)
            
            # Press q if you want to end the loop
            if key == ord('q'):
               break
       
        # #---------------------------------------------------------------------------------------------------------------------------------------
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(hCam)

        # # Destroys the OpenCv windows
        cv2.destroyAllWindows()


    def close(self):
        # Release camera resource
        ret = ueye.is_ExitCamera(self.hCam)


if __name__ == "__main__":
    # When done with the camera you must perform a thorcam.close(),
    # otherwise the script will hang at the end
    cam1 = Thorcam(2)
    cam1.showVid()

    cam1.close()
