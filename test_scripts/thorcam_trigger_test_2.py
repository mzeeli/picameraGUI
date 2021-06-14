import sys
import cv2

sys.path.append("../")
from thorcam import Thorcam 

# Initiate thorcam and set trigger timeout to 1000 seconds
cam1 = Thorcam(1, trigTimeout=100000)

# Set exposure time to 0.01 ms
cam1.setExposureTime(0.02)

for i in range(2, 100, 2):
	print(f"----{i}ms----")
	print("Waiting")
	
	# Capture image
	img = cam1.capImgNow()
	# ~ img = cam1.enableHardwareTrig()
	
	cv2.imshow(f"{i}ms, press 'q' to exit", img)
	cv2.imwrite("./images/{:02d}ms.jpg".format(i), img)
	# ~ cv2.imwrite(f"./images/background.jpg", img)
	
	key = cv2.waitKey(0)
	if key == ord('q'):
		print("Exciting capture loop")
		break
		
	cv2.destroyAllWindows()
	
	
cv2.destroyAllWindows()
cam1.close()
print("Closed")
