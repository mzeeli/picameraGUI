#############################################################################
#############################################################################
# Magneto Optical Trap Time of Flight Analyzer
#
# Updated: Jun. 11, 2018
# Coded by Taehyun Yoon
# Property of the Bajcsy NPQO Group
# Institute for Quantum Computing
# 
# This script followed the method in [Luksch, Thesis, National University of Singapore (2012)]
#############################################################################
#############################################################################

# DEBUGGING FLAGS
importflag = True
centerfindflag = True
gaussianfitflag = True
visualflag = False
Gauss2Dflag = True

fitaxis = 0    # 0: x-axis, 1: y-axis

cropx = 520     # center x position of the crop square 
cropy = 600     # center y position of the crop square
cropsize = 350  # Half of crop square dimension

rcropx = 800    # Crop for background intensity comparison
rcropy = 650
rcropsize = 100 # Half of rcrop square dimension

hbar = 1.0546e-34 # [m^2 kg/s]
Gamma = 5.22  # Natural linewidth [MHz]
omega = 351.722e6 # Probe frequency [MHz]
detuning = 0 * Gamma    # [MHz]
Isat = 1.09 # Saturation intensity [mW/cm^2]
#crossSec0 = 1.0e+11 * hbar * omega * Gamma / (2*Isat) # Cross section on resonance [m^2]
#crossSec = crossSec0 / (1+4*(detuning/Gamma)**2+I0/Isat)
crossSec = 346.9e-15 # [m^2]
probePower = 5.0e-6 # [W]
OD_upper_bound = 4  # valid OD bound value

# IMPORT STATEMENTS
import numpy as np
from PIL import Image, ImageFilter
import glob
from pylab import imshow, show, gray
from matplotlib import pyplot
from matplotlib import mlab
from scipy.optimize import curve_fit
from numpy import linspace

directory = './'
#MOTfiles = ['MOT_1.jpg','MOT_2.jpg','MOT_3.jpg']
MOTfiles = ['04ms.png']
#probefiles = ['probe_1.jpg','probe_2.jpg','probe_3.jpg']
probefiles = ['background.png']
#bgfiles = ['bg_1.jpg','bg_2.jpg','bg_3.jpg']
bgfiles = ['bg.png']

# SIGMA VALUE FINDING STAGE

def imgavr(filelist): # function to fit to sigma array
    imgarray = np.zeros((2*cropsize, 2*cropsize))
    # imgarray = np.zeros((1024, 1280))
    for filename in filelist:
        filename = directory + filename
        imgsingle = np.array(Image.open(filename).convert('L'))
        imgarray = imgarray + imgsingle[cropy-cropsize:cropy+cropsize,cropx-cropsize:cropx+cropsize]
        # imgarray = imgarray + imgsingle
        del imgsingle
    return imgarray/len(filelist)

# sig_ar *= 2.5*0.0053*10**-3 # convert px number to meter
px_meter = 13.7e-6 # ratio of px number to meter

MOTimg = imgavr(MOTfiles)
probeimg = imgavr(probefiles)
bgimg = imgavr(bgfiles)
MOTimg = MOTimg - bgimg
probeimg = probeimg - bgimg

#PtoB = probePower/numpy.sum(probeimg) # ratio of probe power to a brightness unit
#MOTint = MOTimg*PtoB # Intensity array of MOT image
#probeint = probeimg*PtoB # Intensity array of probe image


MOTimg = np.array([[j if j>0 else 1 for j in i] for i in MOTimg])
probeimg = np.array([[j if j>0 else 1 for j in i] for i in probeimg])
ODarray = -np.log(MOTimg/probeimg)
ODarray = np.array([[j if j<OD_upper_bound else 1 for j in i] for i in ODarray])

	
atomNum = (px_meter**2)*np.sum(ODarray)/crossSec
print("Atom Number: {0:.3e}".format(atomNum))

pyplot.imshow(MOTimg, origin='lower', cmap='gray', vmin=0, vmax=255) 
pyplot.axis('off')
pyplot.tight_layout()
pyplot.show()
#pyplot.savefig('MOT_img.eps',dpi=100)
