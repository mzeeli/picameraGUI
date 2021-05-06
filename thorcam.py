"""
Wrapper for pyueye IDS camera functions. Use the Thorcam class to connect 
and adjust the thorcam camera parameters.

The thorcams last used were the UI-3240CP-NIR-GL-TL model.

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
	def __init__(self, camID, width=1280, height=1080, trigTimeout=0):
		"""
		camID: (int [1,2]) determines which camera to open
		trigTimeout: Trigger mode timeout in 10ms.
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
		self.width = 1280
		self.height = 1080
		rect_aoi = ueye.IS_RECT()
		rect_aoi.s32X = ueye.int(0)
		rect_aoi.s32Y = ueye.int(0)
		rect_aoi.s32Width = ueye.int(self.width)
		rect_aoi.s32Height = ueye.int(self.height)
		ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
		
		# set timeout value for trigger mode
		ret = ueye.is_SetTimeout(self.hCam, ueye.IS_TRIGGER_TIMEOUT, trigTimeout)
		
		#################################################################
		# Memory
		#################################################################
		# allocate memory
		self.mem_ptr = ueye.c_mem_p()
		self.mem_id = ueye.int()
		self.nBitsPerPixel = 24 # for colormode = IS_CM_BGR8_PACKED
		ret = ueye.is_AllocImageMem(self.hCam, self.width, self.height, 
									self.nBitsPerPixel, self.mem_ptr, self.mem_id)

		# set active memory region
		ret = ueye.is_SetImageMem(self.hCam, self.mem_ptr, self.mem_id)
		self.lineinc = self.width * int((self.nBitsPerPixel + 7) / 8)


	def set_exposure_time(self, time, debug=False):
		#################################################################
		#set exposure time
		#################################################################
		time_input = ueye.double(time)
		nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_input, 8)
		
		if nRet != ueye.IS_SUCCESS:
			print("is_Exposure Input ERROR")
		
		#################################################################
		# Take image without doing anything to adjust for exposure time
		
		# First image from camera is always set to max exposure time, so
		# a work around was to throw away the first image everytime when 
		# using the camera
		#################################################################
		capImgNow()
		
		
		#################################################################
		# Debugging
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


	def capImgNow(self):
		# 0: instant image, 2: set trigger
		ret = ueye.is_SetExternalTrigger(self.hCam, 0) 
		
		# Wait until 1st image is in memory, timing depends on is_SetExternalTrigger
		ret = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
		img = ueye.get_data(self.mem_ptr, self.width, self.height, 
							self.nBitsPerPixel, self.lineinc, copy=True)
							
		img = np.reshape(img, (self.height, self.width, 3))
		return img


	def enableHardwareTrig(self):
		# 0: instant image, 2: set trigger
		ret = ueye.is_SetExternalTrigger(self.hCam, 2)
		
		# Wait until 1st image is in memory, timing depends on is_SetExternalTrigger
		ret = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
		img = ueye.get_data(self.mem_ptr, self.width, self.height, 
							self.nBitsPerPixel, self.lineinc, copy=True)
							
		img = np.reshape(img, (self.height, self.width, 3))
		return img
		
	def close(self):
		# Release camera resource
		ret = ueye.is_ExitCamera(self.hCam)
		
		# ~ print(f"ExitCamera returns {ret}")
		

if __name__ == "__main__":
	# When done with the camera you must perform a thorcam.close(),
	# otherwise the script will hang at the end
	cam1 = Thorcam(1)
	cam2 = Thorcam(2)

	# Test software image capture
	img1 = cam1.capImgNow()
	img2 = cam2.capImgNow()
	cv2.imshow("img1", img1)
	cv2.imshow("img2", img2)
	cv2.waitKey(0)
	print("Done")
	

	cam1.close()
	cam2.close()
