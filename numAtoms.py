"""
Script to calculate number of atoms given a MOT image
Todo: Should have both fluorescent and absorption imaging techniques

The absorption imaging # atom script followed the method in [Luksch, Thesis,
National University of Singapore (2012)]. This script was originally written by
Taehyun Yoon but for a different setup. I modified it to fit the raspberry pi
system

The fluorescent imaging technique #atom script follows

Last Updated: Winter, 2021
Author: Michael Li
"""

import numpy as np
import os
import re

from matplotlib import pyplot
from PIL import Image
from cesium import Cesium
from motCamera import MOTCamera

cs = Cesium()


def imgavr(filelist, y):  # function to fit to sigma array
    """
    Function included with Taehyun's script

    Reads a list of files in the target directory and averages their values,
    goal is to get an averaged image of the mot to reduce noise

    :param filelist: list of image paths of the same image
    :return: averaged image of the mot
    """
    cropx = 520  # center x position of the crop square, default 520
    cropy = y  # center y position of the crop square, default 600
    cropsize = 350  # Half of crop square dimension

    rcropx = 800  # Crop for background intensity comparison
    rcropy = 650
    rcropsize = 100  # Half of rcrop square dimension

    for filename in filelist:
        filename = filename
        # read image as grayscale
        imgsingle = np.array(Image.open(filename).convert('L'))
        imgarray = imgsingle[cropy - cropsize:cropy + cropsize,
                   cropx - cropsize:cropx + cropsize]
        # imgarray = imgarray + imgsingle
    return imgarray / len(filelist)


def getNumAtomsLegacy(motImgPath, probeImgPath, bgImgPath, y, showImg=False):
    """
    This is the original algorithm Taeyoon wrote to calculate number of atoms
    It followed the method in Luksh's Thesis

    """
    # Constants #
    fitaxis = 0  # 0: x-axis, 1: y-axis

    hbar = 1.0546e-34  # [m^2 kg/s]
    Gamma = 5.22  # Natural linewidth [MHz]
    omega = 351.722e6  # Probe frequency [MHz]
    detuning = 0 * Gamma  # [MHz]
    Isat = 1.09  # Saturation intensity [mW/cm^2]
    # Cross section on resonance [m^2]
    # crossSec0 = 1.0e+11 * hbar * omega * Gamma / (2*Isat)
    # crossSec = crossSec0 / (1+4*(detuning/Gamma)**2+I0/Isat)
    crossSec = 346.9e-15  # [m^2]
    probePower = 5.0e-6  # [W]
    OD_upper_bound = 4  # valid OD bound value, original was 4

    px_meter = 13.7e-6  # ratio of px number to meter

    # Read Images #

    MOTimg = imgavr(motImgPath, y)
    probeimg = imgavr(probeImgPath, y)
    bgimg = imgavr(bgImgPath, y)

    # Subtract background
    MOTimg = MOTimg - bgimg
    probeimg = probeimg - bgimg

    # Calculate optical density by ln(I/I_0) #
    # Motimg can not equal zero otherwise log(0), instead set as 1
    MOTimg = np.array([[j if j > 0 else 1 for j in i] for i in MOTimg])
    # probeimg can not equal zero otherwise division by zero, instead set as 1
    probeimg = np.array([[j if j > 0 else 1 for j in i] for i in probeimg])

    # get optical density, Eq 1.4 in Lusch thesis
    ODarray = -np.log(MOTimg / probeimg)
    print(f"------{motImgPath}------")
    print(f"Highest OD value: {ODarray.max()}")
    ODarray = np.array([[j if j < OD_upper_bound else 1 for j in i]
                        for i in ODarray])

    atomNum = (px_meter ** 2) * np.sum(
        ODarray) / crossSec  # Eq 1.5 in Luksch thesis
    print("Atom Number: {0:.3e}".format(atomNum))

    if showImg:
        pyplot.imshow(MOTimg, origin='lower', cmap='gray', vmin=0, vmax=255)
        pyplot.axis('off')
        pyplot.tight_layout()
        pyplot.show()

    return atomNum


def numAtomsAbs(motImgPath, probeImgPath, bgImgPath, y=600):
    """
    Calculates the number of atoms based on the probe beam absorption method.
    This is the main method outlined in Luksch's Thesis

    :param motImgPath:
    :param probeImgPath:
    :param bgImgPath:
    :return:
    """
    # Constants #

    crossSec = 346.9e-15  # [m^2]
    OD_upper_bound = 6  # valid OD bound value, original was 4

    px_meter = 13.7e-6  # ratio of px number to meter

    # Read Images #
    MOTimg = imgavr(motImgPath, y)
    probeimg = imgavr(probeImgPath, y)
    bgimg = imgavr(bgImgPath, y)

    # Subtract background
    MOTimg = MOTimg - bgimg
    probeimg = probeimg - bgimg

    # Calculate optical density by ln(I/I_0) #
    # Motimg can not equal zero otherwise log(0), instead set as 1
    MOTimg = np.array([[j if j > 0 else 1 for j in i] for i in MOTimg])
    # probeimg can not equal zero otherwise division by zero, instead set as 1
    probeimg = np.array([[j if j > 0 else 1 for j in i] for i in probeimg])

    # get optical density, Eq 1.4 in Lusch thesis
    ODarray = -np.log(MOTimg / probeimg)

    # Set upper bound, if any value in ODarray is > OD_upper_bound set it to 1
    ODarray = np.array([[j if j < OD_upper_bound else 1 for j in i] for i in
                        ODarray])

    # Eq 1.5 in Luksch thesis
    atomNum = (px_meter ** 2) * np.sum(ODarray) / crossSec
    # print("Atom Number: {0:.3e}".format(atomNum))
    #
    # pyplot.imshow(MOTimg, origin='lower', cmap='gray', vmin=0, vmax=255)
    # pyplot.axis('off')
    # pyplot.tight_layout()
    # pyplot.show()

    return atomNum


def numAtomsAbsOverTime(imgDir):
    """
    Performs numAtomsAbs on all image files in a given folder.
    Returns list of #atoms calculated and their respective timestamp

    :param imgDir: Directory containing all image files, see "Absorption images
                   example" for reference of structure

    :return: list of #Atoms for a given image, list of corresponding time
    """

    ## Find Image Files ##
    # Setup Regex conditions
    # Use regex to find img files. Ex: 04ms.png or 06ms.jpg
    motImgCond = re.compile(r'\d{2}ms.(png|jpg)')
    probeImgCond = re.compile(r'background')
    bgImgCond = re.compile(r'bg')

    # Go through files and ind file paths of interest
    for root, dirs, files in os.walk(imgDir):
        # Find all files that satisfy regex conditions
        imgFiles = [os.path.join(imgDir, f)
                    for f in files if re.match(motImgCond, f)]
        probeImgPath = [os.path.join(imgDir, f)
                        for f in files if re.match(probeImgCond, f)]
        bgImgPath = [os.path.join(imgDir, f)
                     for f in files if re.match(bgImgCond, f)]

    # Get time in ms from file names
    imgTimes = [int(re.findall(r'\d{2}', f)[0]) for f in imgFiles]

    atomNum = []
    y = 600
    for i, path in enumerate(imgFiles):
        atomCount = getNumAtomsLegacy([path], probeImgPath, bgImgPath, y)
        atomNum.append(atomCount)
        y = y - 15

    # pyplot.plot(imgTimes, atomNum, '-o')
    # pyplot.xlabel("time (ms)")
    # pyplot.ylabel("#Atoms")
    # pyplot.show()

    return atomNum, imgTimes


def numAtomsFlu():
    """
    Calculates the number of atoms based on the fluorescent method.

    :return: # atoms
    """

    # Camera Hardware Properties
    fStop = 2  # [f/#]
    focalLength = 3.04e-3  # [m]


    # Experimental parameters
    intensity = -1  # [mW/cm^2] TODO get real intensity
    detuning = -1  # [Hz]TODO get real detuing
    distance = -1  # [m] TODO get real distance

    # 
    s = intensity / cesium.I_sat  # aturation parameter
    apertureR = focalLength / (fStop * 2)  # Camera lens aperture radius
    solidAngle = apertureR**2 / (4 * distance**2)  # Fraction of solid angle

    # Equation (2) in "Photo-scattering rate meas of atom in a MOT" paper
    scatteringRate = cs.lineWidth/2 * s/(1+s+(2*detuning/cs.lineWidth)**2)


if __name__ == "__main__":
    # Expect og group to have 80 million
    # Expect mot1 to have 10-20 million
    # Expect mot2 to have 40 million

    # print("--First Set--")
    # motImgPath = ['legacy/Absorption images example/04ms.png']
    # probeImgPath = ['legacy/Absorption images example/background.png']
    # bgImgPath = ['legacy/Absorption images example/bg.png']
    # n = numAtomsAbs(motImgPath, probeImgPath, bgImgPath)
    # print(n)

    cesium = Cesium()
