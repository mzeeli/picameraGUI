"""
Script to calculate MOT temperature given a series of absorption images

The TOF temperature calculation script follows "Measurement of Temperature of
Atomic Cloud Using Time-of-Flight Technique" by P. Arora et al.

This script is built from Turner Silverthorne's MOTTOF3.py script but
for a different setup. I cleaned it up and modified it to fit the raspberry pi
system.

HOW TO USE:
Again the general algorithm wasn't written by me but these are some of the key
points I picked up while using the script:

1. First make sure the ROIs in image_to_sigma are centered correctly
2. If image is dark make sure the threshold var in image_to_sigma is set to
lower
3. The image can't have saturated or 0 DN values in the ROIs
4. The 2D gaussian in image_to_sigma is a little buggy

5. Previously the script could run with the gaussian fit axis in both x and
y. However a change was made to use the gravitation constant as calibration,
so the only allowed axis of calibration is now y because we need the y pos
of the mot in each frame


Last Updated: Summer Term, 2021
Author: Michael Li
"""

import os
import re
import cv2
import numpy as np
import warnings

from PIL import Image
from pylab import gray
from matplotlib import pyplot
from scipy.optimize import curve_fit
from math import sqrt, exp
from cesium import Cesium

cesium = Cesium()

# sometimes image_to_sigma takes a long time and raises an annoying warning
# this just suppresses that specific warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def findImgFiles(imgDir):
    """
    Finds necessary image files in the imgDir directory for temperature calculations.

    Finds and returns images titled similar to '##ms.png' and the
    'background.png' file path

    :param imgDir: (str: filepath) Path to folder containing mot images. Ex.
                   imgDir=r'./legacy/Absorption image example'
    :return: MOT image directory paths as a list, path to probe image
    """
    # Use regex to find img files. Ex: 04ms.png or 06ms.jpg
    motImgCond = re.compile(r'\d{2}ms.(png|jpg)')
    probeImgCond = re.compile(r'background.(png|jpg)')

    for root, dirs, files in os.walk(imgDir):
        # Find all files that satisfy regex conditions
        imgFiles = [os.path.join(imgDir, f)
                    for f in files if re.match(motImgCond, f)]

        probeImgFile = [os.path.join(imgDir, f)
                        for f in files if re.match(probeImgCond, f)][0]
    imgFiles.sort()
    return imgFiles, probeImgFile


def gaussianRadiiFunc(t, sigma_0, sigma_v):
    """
    Equation to represent gaussian cloud radii at a given time t

    Use this function to determine sigma_v with curve_fit

    This is the equation just below equation (4) of "Measurement of
    Temperature of Atomic Cloud Using Time-of-Flight Technique"
    et al.

    :param t: (int) time [s]
    :param sigma_0: (double) initial radii of cloud [m]
    :param sigma_v: (double) speed of cloud radii increase [m/s]
    :return: (double) gaussian raddi function's evaluation
    """

    return np.sqrt(sigma_0**2 + sigma_v**2 * t**2)


def kinematicEqFunc(t, v_i, px2Meter, x0):
    """
    Kinematic equation of motion equation

    Can be used to determine a pixel to meter value that corresponds to the
    camera's depth of field while using a constant gravity 9.81 m/s^2 as a
    reference.

    :param t: Time [s]
    :param v_i: Initial velocity [m/s]
    :param px2Meter: Camera pixel to meter conversion [m/pixel]
    :param x0: Initial pixel position
    :return: Position in units of camera pizels
    """
    # Note 9.81 is gravitational acceleration
    return (v_i * t - 0.5 * 9.81 * t ** 2 + x0) / px2Meter

def getTemperature(mass, sigma_v):
    """
    Equation to get temperature of MOT

    Based on equation (2) in "Measurement of Temperature of Atomic Cloud
    using Time-of-flight Technique" paper

    :param mass: (double) Atomic mass [kg]
    :param sigma_v: (double) rate of cloud radii increase [m/s]
    :return: (double) Temperature [K]
    """
    k_b = 1.38e-23  # Boltzman Constant
    T = mass / k_b * sigma_v**2
    return T


def image_to_sigma(imgback, imgfore, roiflag=False, visualflag=False,
                   Gauss2Dflag=False):
    """
    Gets cloud radii for a given absorption image

    This function was originally a script written by Turner Silverthorne,
    I didn't change the algorithm, just refactored it and made it modular.

    I'm not too familiar with how the Gauss2D works, only the 1D works
    reliably in my experience and I've been mostly modifying the 1D

    :param imgback: (2D np.array) Background image of just probe laser
    :param imgfore: (2D np.array) Image of MOT with probe laser

    :param roiflag: (bool) toggles ROI cropping visuals for debugging
    :param visualflag: (bool) toggles debugging visuals
    :param Gauss2Dflag: (bool) toggles between 1D or 2D gaussian fitting

    :return: (double) Cloud radii sigma_t in units of pixels, (int) center
    position of mot
    """
    # constants
    fitaxis = 1  # 0: x-axis, 1: y-axis. Must be fit on y-axis to use gravity

    ##############################################################
    ### ROI for absorption images on Pi
    rcropx = 930  # Crop for background intensity comparison, 460
    rcropy = 380
    rcropsize = 300  # Half of rcrop square dimension

    cropx = 930  # center x position of the crop square, 470
    cropy = 380  # center y position of the crop square
    cropsize = 350  # Half of crop square dimension
    ##############################################################

    ##############################################################
    ### ROI for absorption images from Turner
    # cropx = 470  # center x position of the crop square
    # cropy = 400  # center y position of the crop square
    # cropsize = 350  # Half of crop square dimension
    #
    # rcropx = 460  # Crop for background intensity comparison
    # rcropy = 480
    # rcropsize = 300  # Half of rcrop square dimension


    ##############################################################
    # to be used in offsetting background from MOT image
    # you can tune them if you want, but I usually leave them at 0
    delx = 0
    dely = 0

    ##############################################################
    # Crop the images and take their difference
    imgBackRatioROI = imgback[rcropy - rcropsize:rcropy + rcropsize,
                              rcropx - rcropsize:rcropx + rcropsize]
    imgForeRatioROI = imgfore[
                      rcropy - rcropsize + dely:rcropy + rcropsize + dely,
                      rcropx - rcropsize + delx:rcropx + rcropsize + delx]

    imgBackDifROI = imgback[cropy - cropsize:cropy + cropsize,
                            cropx - cropsize:cropx + cropsize]
    imgForeDifROI = imgfore[cropy - cropsize + dely:cropy + cropsize + dely,
                            cropx - cropsize + delx:cropx + cropsize + delx]

    ratio = np.sum(imgBackRatioROI) / np.sum(imgForeRatioROI)
    imdif = np.array(imgBackDifROI - ratio * imgForeDifROI, 'float')
    (X, Y) = np.shape(imdif)

    if roiflag:
        pyplot.subplot(2, 3, 1)
        pyplot.imshow(imgBackRatioROI, cmap='gray', vmin=0, vmax=255)

        pyplot.subplot(2, 3, 2)
        pyplot.imshow(imgForeRatioROI, cmap='gray', vmin=0, vmax=255)

        pyplot.subplot(2, 3, 4)
        pyplot.imshow(imgBackDifROI, cmap='gray', vmin=0, vmax=255)

        pyplot.subplot(2, 3, 5)
        pyplot.imshow(imgForeDifROI, cmap='gray', vmin=0, vmax=255)

        pyplot.subplot(2, 3, 6)
        pyplot.imshow(imdif, cmap='gray', vmin=0, vmax=255)

        pyplot.tight_layout()
        pyplot.show()

    ##############################################################
    # find center of brightness
    # loop to find center of image
    thresh = 15  # Old value was 25
    m = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            m[x, y] = imdif[x, y] >= thresh

    m = m / np.sum(m)

    # marginal distributions as percentage
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values by weighted sum
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))

    # optional visualization
    if visualflag:
        pyplot.figure(1, figsize=(4, 6.5))
        pyplot.subplot(211)
        pyplot.imshow(imdif)
        if fitaxis == 0:
            pyplot.plot([0, 2 * cropsize], [cx, cx], 'b-', lw=2)
        elif fitaxis == 1:
            pyplot.plot([cy, cy], [0, 2 * cropsize], 'b-', lw=2)

        pyplot.plot(cy, cx, 'bo')
        gray()
        pyplot.gcf()
    ##############################################################
    # take horizontal and vertical cuts of I_bck - I_mot and also I_bck
    imref_full = np.array(imgback[cropy - cropsize:cropy + cropsize,
                                  cropx - cropsize:cropx + cropsize], 'float')
    if fitaxis == 0:
        imref_cut = np.array(imref_full[int(cx), :], 'float')
        samp = np.array(imdif[int(cx), :], 'float')
    elif fitaxis == 1:
        imref_cut = np.array(imref_full[:, int(cy)], 'float')
        samp = np.array(imdif[:, int(cy)], 'float')

    x = np.arange(len(samp))
    z = np.zeros(len(samp))

    # take log only at pixel values where log is well defined
    for i in range(len(samp)):
        if samp[i] / imref_cut[i] < 1 and imref_cut[i] > 0 and samp[i] > 0:
            z[i] = -np.log(1.0000 - samp[i] / imref_cut[i])

    # find average and sigma
    mean = sum(x * z) / sum(z)
    sigma = np.sqrt(sum(z * (x - mean) ** 2) / sum(z))
    ambient = 0.01

    ##############################################################
    # Perform fitting
    if Gauss2Dflag:
        zz = np.zeros([X, Y])
        for i in range(X):
            for j in range(Y):
                if imdif[i, j] / imref_full[i, j] and imdif[i, j] > 0:
                    zz[i, j] = -np.log(1.0000 - imdif[i, j] / imref_full[i, j])
        zz1d = zz.ravel()

        def Gauss2D(XX, a, x0, y0, sigma, b):
            val = a * np.exp(
                -((XX[0] - x0) ** 2 + (XX[1] - y0) ** 2) / (2 * sigma ** 2)) + b
            return val.ravel()

        # create x and y indices
        xx = np.linspace(0, X - 1, X)
        yy = np.linspace(0, Y - 1, Y)
        xx, yy = np.meshgrid(xx, yy)

        # Define boundaries for fitting parameters
        #                    a, x0   y0  sigma         b
        boundaries = ((-np.inf,  0,   0,     0,  -np.inf),
                      (np.inf,   X,   Y,   X/2,   np.inf))

        popt, pcov = curve_fit(Gauss2D, (xx, yy), zz1d, bounds=boundaries, 
                               p0=[np.amax(zz), cx, cy, sigma, ambient])

        # xLinspace = np.linspace(0, X - 1, X)
        # yar = Gauss2D([xLinspace, 520], popt[0], popt[1], popt[2], popt[3],
        #               popt[4])
        #
        # pyplot.plot(xLinspace, zz[520, :], 'ko', ms=5)
        # pyplot.plot(xLinspace, yar, 'b-', lw=3)
        #
        # pyplot.tight_layout()
        # pyplot.show()


    else:
        # define curve to be fitted
        def Gauss(x, a, x0, sigma, b):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b

        # Define boundaries for fitting parameters
        #                    a,  x0  sigma         b
        boundaries = ((-np.inf,   0,     0,  -np.inf),
                      ( np.inf,   X,   X/2,   np.inf))
        # fit curve
        popt, pcov = curve_fit(Gauss, x, z, bounds=boundaries, p0=[max(z), mean, sigma, 0.01])

        # optional visualization
        if visualflag:
            pyplot.figure(1)
            pyplot.subplot(212)

            pyplot.plot(x, z, 'b+:', label='data')
            pyplot.plot(x, Gauss(x, *popt), 'r-', label='fit')
            pyplot.legend()

            pyplot.title('Gaussian Fit Trial')
            pyplot.xlabel('Pixel Number')
            pyplot.ylabel('Intensity 0 - 255')
            pyplot.gcf
            pyplot.show()

    ##############################################################
    # Print final sigma results
    if Gauss2Dflag:
        # print('sigma = {0:.6f},\t popt[3] = {1:.6f}'.format(sigma, popt[3]))
        return popt[3]

    else:
        # print('sigma = {0:.6f},\t popt[2] = {1:.6f}'.format(sigma, popt[2]))
        return popt[2], popt[1]  # For 1D also return the center position


def getTempFromImgList(filelist, bgImgPath, showSigmaFit=False, verifyG=False):
    """
    Calculates and returns temperature based on a series of absorption images

    :param filelist: (str) List of Image paths
    :param bgImgPath: (str) image path to background image

    :param showSigmaFit: (bool) toggles view of pyplot fit
    :param verifyG: (bool) toggles view gravitational constant verification (
    note only makes sense if fit axis is 1 (i.e. the y direction)

    :return T: (np.float64) temperature of MOT cloud [Kelvin]
    """

    sig_ar = np.zeros(len(filelist))  # Array to store the sigma values
    t_ar = np.zeros(len(filelist))  # time array
    yCenters = np.zeros(len(filelist))  # Y center positions in picture

    bgarray = np.array(Image.open(bgImgPath).convert('L'))

    for index, filename in enumerate(filelist):
        # Use regex to find time when image was taken
        fileTime = re.findall(r'\d{2}', filename)[-1]
        print(f"{fileTime}ms \t {filename}")
        t_ar[index] = float(fileTime)

        imgarray = np.array(Image.open(filename).convert('L'))
        sig_ar[index], yCenters[index] = image_to_sigma(bgarray, imgarray,
                                                        roiflag=False,
                                                        visualflag=False)

    ###########################################################################
    # Calibrate px2Meter value using gravitational constant
    print("\n----------------Calibration----------------")

    t_ar *= 1e-3  # convert time to ms

    popt, _ = curve_fit(kinematicEqFunc, t_ar, yCenters)
    
    # Fit param #2 of kinematicEqFunc is px2Meter val
    px2Meter = popt[1]  # Old hard coded val in Turner's script was 13.7e-6
    print('Pixel to meter conversion: {:0.4e} m/px'.format(px2Meter))

    if verifyG:
        time_linspace = np.linspace(0, max(t_ar), 1000)
        yar = kinematicEqFunc(time_linspace, popt[0], popt[1], popt[2])

        pyplot.plot(1000 * t_ar, yCenters, 'ko', ms=5)
        pyplot.plot(1000 * time_linspace, yar, 'b-', lw=3)
        pyplot.legend(["Y Center Position", "Kinematic Equation Fit"],
                      fontsize=12)
        pyplot.title('Grav Constant Calibration')
        pyplot.xlabel('time [ms]', fontsize=18)
        pyplot.ylabel('MOT Y Center [px]', fontsize=18)
        pyplot.xticks(np.arange(0, 26, 2), fontsize=10)
        pyplot.tight_layout()
        pyplot.show()
    ###########################################################################
    # LINEAR REGRESSION/TEMPERATURE OUTPUT STAGE
    sig_ar *= px2Meter  # convert px number to meter

    # Calculate sigma values
    # popt[0] = sigma_o, popt[1] = sigma_v
    popt, pcov = curve_fit(gaussianRadiiFunc, t_ar, sig_ar)  # curve fitting
    sigma_o = popt[0]
    sigma_v = popt[1]

    # Calculate R^2 (coefficient of determination)
    residuals = sig_ar - gaussianRadiiFunc(t_ar, sigma_o, sigma_v)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig_ar - np.mean(sig_ar)) ** 2)
    r_squ = 1 - (ss_res / ss_tot)
    print("\n----------------Temp----------------")
    print('Coeff of corr: {:0.4}'.format(r_squ))

    # Temperature calculation
    M = cesium.atomicMass
    T = getTemperature(M, sigma_v)
    T_microKelvin = T*1e6
    print('Temperature: {:0.4} uK'.format(T_microKelvin))

    if showSigmaFit:
        # Plot fit results with data points for debugging
        v_ar = np.linspace(min(t_ar), max(t_ar), 1000)
        yar = gaussianRadiiFunc(v_ar, sigma_o, sigma_v)

        pyplot.plot(1000 * t_ar, sig_ar, 'ko', ms=5)
        pyplot.plot(1000 * v_ar, yar, 'b-', lw=3)
        pyplot.legend(["Gaussian size", "Temperature fit"], fontsize=12)
        pyplot.title('Temperature to Sigma Fit')
        pyplot.xlabel('time [ms]', fontsize=18)
        pyplot.ylabel('$\sigma_t$', fontsize=18)
        pyplot.xticks(np.arange(0, 26, 2), fontsize=10)
        pyplot.yticks([])
        pyplot.tight_layout()
        pyplot.show()

    return T


if __name__ == "__main__":
    # imgPaths, bgPath = findImgFiles(r"saved_images/MOT2")
    # imgPaths = imgPaths[:-2]

    # imgPaths, bgPath = findImgFiles(r"C:\Users\Michael\PycharmProjects\picameraGUI\legacy\Absorption images example")
    # imgPaths = imgPaths[:-3]  # Ignore last couple of images


    imgPaths, bgPath = findImgFiles(r"/home/pi/Desktop/picameraGUI/test_scripts/images/May 18 - Absorption 3")
    imgPaths = imgPaths[:-1]  # Ignore last couple of images


    getTempFromImgList(imgPaths, bgPath, showSigmaFit=True, verifyG=True)
