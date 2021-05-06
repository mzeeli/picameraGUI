"""
Script used to test responsiveness of thorcam when sent an external trigger.

Test done by shinning light into the vacuum chamber. When light is going through
the chamber the fiber appears bright, and if the timing is off then the fiber 
appears dim. We use this to determine what the appropriate time delay for the
thorcam is on the main lab computer

Last Updated: Summer Term, 2021
Author: Michael Li
"""

from pyueye import ueye
import numpy as np
import cv2

#################################################################
# Camera Parameters
#################################################################
# init camera
hCam = ueye.HIDS(1) # [1,2] 
ret = ueye.is_InitCamera(hCam, None)

# set color mode
ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)

# set region of interest
width = 1280
height = 1080
rect_aoi = ueye.IS_RECT()
rect_aoi.s32X = ueye.int(0)
rect_aoi.s32Y = ueye.int(0)
rect_aoi.s32Width = ueye.int(width)
rect_aoi.s32Height = ueye.int(height)
ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

#################################################################
# Memory
#################################################################
# allocate memory
mem_ptr = ueye.c_mem_p()
mem_id = ueye.int()
nBitsPerPixel = 24 # for colormode = IS_CM_BGR8_PACKED
ret = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, mem_ptr, mem_id)

# set active memory region
ret = ueye.is_SetImageMem(hCam, mem_ptr, mem_id)

def main():
    # set exposure time in ms
    set_exposure_time(60, False)
    
    #################################################################
    # Take image without doing anything to adjust for exposure time
    
    # First image from camera is always set to max exposure time, so
    # a work around was to throw away the first image everytime when 
    # using the camera
    #################################################################
    ret = ueye.is_SetExternalTrigger(hCam, 0)
    ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    lineinc = width * int((nBitsPerPixel + 7) / 8)
    img = ueye.get_data(mem_ptr, width, height, nBitsPerPixel, lineinc, copy=True)

    
    #################################################################
    # Continuously wait for trigger
    #################################################################
    while True:
        # Enable trigger mode memory
        print("----Waiting for external trigger----")
        
        ret = ueye.is_SetExternalTrigger(hCam, 2)  # 0: instant get image, 2: set trigger
        ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
        
        img = ueye.get_data(mem_ptr, width, height, nBitsPerPixel, lineinc, copy=True)
        img = np.reshape(img, (height, width, 3))
        cv2.imshow('Trigger Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    ret = ueye.is_ExitCamera(hcam)
    print(f"ExitCamera returns {ret}")


def set_exposure_time(time, debug=False):
    #################################################################
    #set exposure time
    #################################################################
    time_input = ueye.double(time)
    nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_input, 8)
    
    if nRet != ueye.IS_SUCCESS:
        print("is_Exposure Input ERROR")
    
    #################################################################
    # Debug
    #################################################################
    if debug:
        #get exposure time range
        time_min = ueye.double(0)
        nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, time_min, 8)
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Min. Range ERROR")
        print("Get Exposure Time Min.", time_min)  
         
        time_max = ueye.double(0)
        nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, time_max, 8)
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Max. Range ERROR")
        print("Get Exposure Time Max.", time_max)   

        # Verify newly set exposure time
        time_output = ueye.double()
        nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, time_output, 8)
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Output ERROR")
        print("Get Exposure Time", time_output)       
        
        # Check default exposure time
        time_default = ueye.double()
        nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_DEFAULT, time_default, 8)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")
            
        print("Get Default Exposure Time", time_default)  


if __name__ == '__main__':
    main()
