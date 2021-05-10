"""
Script to calculate MOT temperature given a series of absorption images

The TOF temperature calculation script follows "Measurement of Temperature of
Atomic Cloud Using Time-of-Flight Technique" by P. Arora et al.

This script is loosely built from Turner Silverthorne's MOTTOF3.py script but
for a different setup. I cleaned it up and modified it to fit the raspberry pi
system.


Last Updated: Summer Term, 2021
Author: Michael Li
"""

import os
import re
import cv2
import numpy as np

from PIL import Image
from pylab import gray
from matplotlib import pyplot
from scipy.optimize import curve_fit
from math import sqrt, exp
from cesium import Cesium

cesium = Cesium()


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

    :param imgback: (2D np.array) Background image of just probe laser
    :param imgfore: (2D np.array) Image of MOT with probe laser

    :param roiflag: (bool) toggles ROI cropping visuals for debugging
    :param visualflag: (bool) toggles debugging visuals
    :param Gauss2Dflag: (bool) toggles between 1D or 2D gaussian fitting

    :return: (double) Cloud radii (sigma_t) in units of pixels
    """
    # constants
    fitaxis = 0  # 0: x-axis, 1: y-axis

    rcropx = 460  # Crop for background intensity comparison, 460
    rcropy = 480
    rcropsize = 300  # Half of rcrop square dimension

    cropx = 470  # center x position of the crop square, 470
    cropy = 400  # center y position of the crop square
    cropsize = 350  # Half of crop square dimension

    # to be used in offsetting background from MOT image
    # you can tune them if you want, but I usually leave them at 0
    delx = 0
    dely = 0

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

    # find center of brightness
    # loop to find center of image
    thresh = 25
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
        pyplot.figure(1, figsize=(4, 8))
        pyplot.subplot(211)
        pyplot.imshow(imdif)
        if fitaxis == 0:
            pyplot.plot([0, 2 * cropsize], [cx, cx], 'b-', lw=2)
        elif fitaxis == 1:
            pyplot.plot([cy, cy], [0, 2 * cropsize], 'b-', lw=2)

        pyplot.plot(cy, cx, 'bo')
        gray()
        pyplot.gcf()

    # take horizontal and vertical cuts of I_bck - I_mot
    # and also I_bck
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

        popt, pcov = curve_fit(Gauss2D, (xx, yy), zz1d, bounds=boundaries)

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
        popt, pcov = curve_fit(Gauss, x, z, bounds=boundaries)

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

    # Print final sigma results
    if Gauss2Dflag:
        # print('sigma = {0:.6f},\t popt[3] = {1:.6f}'.format(sigma, popt[3]))
        return popt[3]

    else:
        # print('sigma = {0:.6f},\t popt[2] = {1:.6f}'.format(sigma, popt[2]))
        return popt[2]


def imageToSigmaE2(imgback, imgfore, time=-1, debug=False):
    """
    new version of image_to_sigma I wrote, problem is the image_to_sigma
    function Turner wrote sometimes gives a wrong value when he does curve_fit.

    This function essentially does the same thing as image_to_sigma,
    but instead it determines the radius of the MOT by the 1/e^2 convention

    :param imgback: (2D np.array) image of just the probelaser background
    :param imgfore: (2D np.array) image of the MOT absorbing in front of camera

    :param time: time of the given image, doesn't have to be included but
                 useful information to have with debugging images
    :param debug: (bool) toggles debugging mode to view ROIs in CV2 and also
                  saves the images to "./saved_images/MOT_Temperature/{sigma}.jpg"


    :return: (double) Cloud radii (sigma_t) in units of pixels
    """
    # constants
    fitaxis = 0  # 0: x-axis, 1: y-axis

    cropx = 600  # motCenter x position of the crop square, 470
    cropy = 420  # motCenter y position of the crop square
    cropsize = 350  # Half of crop square dimension

    imgBackROI = imgback[cropy - cropsize: cropy + cropsize,
                         cropx - cropsize: cropx + cropsize]
    imgForeROI = imgfore[cropy - cropsize: cropy + cropsize,
                         cropx - cropsize: cropx + cropsize]

    # "Invert" ROI images so it's fluorescent based

    # Perform background subtraction
    motROI = cv2.subtract(imgBackROI, imgForeROI)

    # Remove noise image with small gaussian blur
    motROI = cv2.GaussianBlur(motROI, (5, 5), cv2.BORDER_DEFAULT)

    # Create mask based on 1/e^2 convention or FWHM convention
    maxIntensity = motROI.max() / exp(2)  # 1/e^2
    ret, mask = cv2.threshold(motROI, maxIntensity, 255, cv2.THRESH_BINARY)

    # Get the largest contour around the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE)   
    c = max(contours, key=cv2.contourArea)  # get largest contour

    # Largest Inner Circle ##
    # Redraw mask with contour so there aren't any "holes" in the mask
    contourMask = np.zeros((mask.shape[1], mask.shape[0])).astype('uint8')
    cv2.drawContours(contourMask, [c], -1, 255, -1)

    distTransform = cv2.distanceTransform(contourMask, cv2.DIST_L2, 3)

    # maxVal = 1/e^2 radius, maxLoc = center position of mot
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(distTransform)
    motCenter = (int(maxLoc[0]), int(maxLoc[1]))
    w = int(maxVal)

    ## Smallest Enclosing Circle Method (abandonned) ##
    # (x, y), radius = cv2.minEnclosingCircle(c)
    # motCenter = (int(x), int(y))
    # radius = int(radius)

    # Convert from 1/e^2 to FWHM radius to calculate sigma
    FWHM_radius = int(2*w/1.699)
    sigma = FWHM_radius / (2*sqrt(2*np.log(2)))

    if debug:
        # Create copies of ROI to draw on them without reprecussion
        motROI_Debug = motROI.copy()
        imgBackROI_Debug = imgBackROI.copy()
        imgForeROI_Debug = imgForeROI.copy()
        imgForeROI_Debug2 = imgForeROI.copy()
        motROI_Debug2 = motROI.copy()

        # Draw contours
        cv2.drawContours(imgBackROI_Debug, c, -1, 255, 2)
        cv2.drawContours(imgForeROI_Debug, c, -1, 255, 2)
        cv2.drawContours(motROI_Debug, c, -1, 255, 2)

        # Draw circle representing MOT estimate
        cv2.circle(motROI_Debug, motCenter, w, 255, 2)
        cv2.circle(mask, motCenter, w, 0, 2)
        cv2.circle(imgForeROI_Debug2, motCenter, w, 255, 2)
        cv2.circle(motROI_Debug2, motCenter, w, 255, 2)

        # Write text labels on the images
        xPos = 10
        yPos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2

        cv2.putText(imgBackROI_Debug, "Background", (xPos, yPos), font,
                    fontScale, 255, thickness)
        cv2.putText(imgBackROI_Debug, f"{time} ms", (xPos, yPos+40), font,
                    fontScale, 255, thickness)
        cv2.putText(imgForeROI_Debug, "Foreground", (xPos, yPos), font,
                    fontScale, 255, thickness)
        cv2.putText(motROI_Debug, "Background-Foreground", (xPos, yPos), font,
                    fontScale, 255, thickness)
        cv2.putText(mask, "Mask", (xPos, yPos), font,
                    fontScale, 255, thickness)
        cv2.putText(imgForeROI_Debug2, "Foreground + MOT Estimate",
                    (xPos, yPos), font, fontScale, 255, thickness)
        cv2.putText(motROI_Debug2, "Background-Foreground + MOT Estimate",
                    (xPos, yPos), font, fontScale, 255, thickness)

        # Create image by stacking all images
        rawROI = np.hstack((imgBackROI_Debug, imgForeROI_Debug, motROI_Debug))
        processedROI = np.hstack((mask, imgForeROI_Debug2, motROI_Debug2))
        debugWindow = np.vstack((rawROI, processedROI))

        cv2.namedWindow('debugWindow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('debugWindow', 1200, 900)
        cv2.imshow("debugWindow", debugWindow)
        writeFile = f"./saved_images/MOT_temperature_debug/{time}-{int(sigma)}"\
                    f".jpg"
        cv2.imwrite(writeFile, debugWindow)
        cv2.waitKey(0)

    return sigma


def getTempFromImgList(filelist, bgImgPath, showSigmaFit=False):
    """
    Calculates and returns temperature based on a series of absorption images

    :param filelist: (str) List of Image paths
    :param bgImgPath: (str) image path to background image
    :param showSigmaFit: (bool) toggles view of pyplot fit

    :return T: (np.float64) temperature of MOT cloud [Kelvin]
    """

    sig_ar = np.zeros(len(filelist))  # Array to store the sigma values
    t_ar = np.zeros(len(filelist))  # time array

    bgarray = np.array(Image.open(bgImgPath).convert('L'))

    for index, filename in enumerate(filelist):
        # Use regex to find time when image was taken
        fileTime = re.findall(r'\d{2}', filename)[0]
        t_ar[index] = float(fileTime)

        imgarray = np.array(Image.open(filename).convert('L'))
        # sig_ar[index] = imageToSigmaE2(bgarray, imgarray, t_ar[index])
        sig_ar[index] = image_to_sigma(bgarray, imgarray)

    # LINEAR REGRESSION/TEMPERATURE OUTPUT STAGE
    sig_ar *= 13.7 * 10 ** -6  # convert px number to meter
    t_ar *= 10 ** -3
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

    print('Coeff of corr: {0}'.format(r_squ))

    # Temperature calculation
    M = cesium.atomicMass
    T = getTemperature(M, sigma_v)
    print('Temperature: {0}'.format(T))

    if showSigmaFit:
        # Plot fit results with data points for debugging
        v_ar = np.linspace(min(t_ar), max(t_ar), 1000)
        yar = gaussianRadiiFunc(v_ar, sigma_o, sigma_v)

        pyplot.plot(1000 * t_ar, sig_ar, 'ko', ms=5)
        pyplot.plot(1000 * v_ar, yar, 'b-', lw=3)
        pyplot.legend(["Gaussian fits", "Temperature fit"], fontsize=16)
        pyplot.title('')
        pyplot.xlabel('time [ms]', fontsize=30)
        pyplot.ylabel('$\sigma_t$', fontsize=30)
        pyplot.xticks(np.arange(0, 26, 2), fontsize=20)
        pyplot.yticks([])
        pyplot.tight_layout()
        pyplot.show()

    return T


if __name__ == "__main__":
    # imgPaths, bgPath = findImgFiles(r"saved_images/MOT2")
    # imgPaths = imgPaths[:-2]

    imgPaths, bgPath = findImgFiles(r"legacy/Absorption images example")
    imgPaths = imgPaths[:-3]  # Ignore last couple of images

    # for i, imgPath in enumerate(imgPaths):
    #     img = cv2.imread(imgPath, 0)
    #     img = img[0:750, 360:740]
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     outDir = "C:\\Users\\Michael\\OneDrive\\Co-op 5\\NPQO\\Weekly " \
    #              "Presentations\\Mine\\"
    #
    #     cv2.imwrite(f"{outDir}\\{i}.png", img)

    getTempFromImgList(imgPaths, bgPath, showSigmaFit=False)
