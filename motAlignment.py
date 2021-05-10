"""
Calculate mot position given two pi camera images

The Pi cameras are expected to be placed at roughly 90 degrees from each 
other. This way we can measure the offset in both dimensions. Otherwise a rotation
matrix needs to be applied.

Last Updated: Summer Term, 2021
Author: Michael Li
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROIs for when both cameras are in front
# Determined these values just based on trial and error
cam0xROI = 250
cam0yROI = 160
cam1xROI = 225
cam1yROI = 170
lengthROI = 40


def getFiberMOTDistance(cam0MotImgRaw, cam1MotImgRaw, debug=False):
    """
    given two picamera images calculates relative distance between mot and fiber

    Important: Both images must be in grayscale

    :param cam0MotImgRaw: (2D np.array) Raw uncropped grayscale camera 0 mot image
    :param cam1MotImgRaw: (2D np.array) Raw uncropped grayscale camera 1 mot image

    :param debug: (bool) debugging flag

    :return: (float tuple) x, y, z of the mot position in camera pixels
    """

    ################################################################
    # Initial ROI
    ################################################################
    cam0MotROI = cam0MotImgRaw[cam0yROI : cam0yROI + lengthROI,
                               cam0xROI : cam0xROI + lengthROI]

    cam1MotROI = cam1MotImgRaw[cam1yROI : cam1yROI + lengthROI,
                               cam1xROI : cam1xROI + lengthROI]

    ################################################################
    # Get fiber's location
    ################################################################
    # Get x position of fiber
    fiberx0_roi, fibery0_roi = getFiberCenter(cam0MotROI)
    fiberx1_roi, fibery1_roi = getFiberCenter(cam1MotROI)

    fiberx0_global = fiberx0_roi + cam0xROI
    fibery0_global = fibery0_roi + cam0yROI
    fiberx1_global = fiberx1_roi + cam1xROI
    fibery1_global = fibery1_roi + cam1yROI


    ################################################################
    # Get MOT's location
    ################################################################
    # crop roi from the tip of the fiber
    cam0MotROI_fiberless = cam0MotROI[:fibery0_roi, :]
    cam1MotROI_fiberless = cam1MotROI[:fibery1_roi, :]

    # ~ if debug:
        # ~ cv2.namedWindow('cam0MotROI_fiberless', cv2.WINDOW_NORMAL)
        # ~ cv2.resizeWindow('cam0MotROI_fiberless', 800, 800)
        # ~ cv2.namedWindow('cam1MotROI_fiberless', cv2.WINDOW_NORMAL)
        # ~ cv2.resizeWindow('cam1MotROI_fiberless', 800, 800)
        # ~ cv2.imshow("cam0MotROI_fiberless", cam0MotROI_fiberless)
        # ~ cv2.imshow("cam1MotROI_fiberless", cam1MotROI_fiberless)
        # ~ cv2.waitKey(0)

    x0_roi, y0_roi, _ = getMOTCenter(cam0MotROI_fiberless)
    x1_roi, y1_roi, _ = getMOTCenter(cam1MotROI_fiberless)

    # overall pixel location outside of ROI
    x0_global = x0_roi + cam0xROI
    y0_global = y0_roi + cam0yROI
    x1_global = x1_roi + cam1xROI
    y1_global = y1_roi + cam1yROI

    relativeX = x0_global - fiberx0_global
    relativeY = x1_global - fiberx1_global
    relativeZ = ((fibery0_global-y0_global)+(fibery1_global-y1_global))/2

    ################################################################
    # Debugging
    ################################################################
    if debug:
        cv2.circle(cam0MotImgRaw, (x0_global, y0_global), 5, 255, 1)
        cv2.circle(cam1MotImgRaw, (x1_global, y1_global), 5, 255, 1)
        cv2.line(cam0MotImgRaw, (fiberx0_global, 0), (fiberx0_global, 272), (255, 0, 255), 1)
        cv2.line(cam1MotImgRaw, (fiberx1_global, 0), (fiberx1_global, 272), (255, 0, 255), 1)
        cv2.line(cam0MotImgRaw, (0, fibery0_global), (544, fibery0_global), (255, 0, 255), 1)
        cv2.line(cam1MotImgRaw, (0, fibery1_global), (544, fibery1_global), (255, 0, 255), 1)

        cv2.rectangle(cam0MotImgRaw, (cam0xROI, cam0yROI),
                      (cam0xROI+lengthROI, cam0yROI+lengthROI),
                      255, 2)
        cv2.rectangle(cam1MotImgRaw, (cam1xROI, cam1yROI),
                      (cam1xROI+lengthROI, cam1yROI+lengthROI),
                      255, 2)

        motCombinedRaw = np.vstack((cam0MotImgRaw, cam1MotImgRaw))

        # ~ cv2.namedWindow('MotImgRaw', cv2.WINDOW_NORMAL)
        # ~ cv2.resizeWindow('MotImgRaw', 800, 800)
        # ~ cv2.imshow("MotImgRaw", motCombinedRaw)

        cv2.namedWindow('cam0MotROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam0MotROI', 400, 400)
        cv2.namedWindow('cam1MotROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam1MotROI', 400, 400)
        cv2.imshow("cam0MotROI", cam0MotROI)
        cv2.imshow("cam1MotROI", cam1MotROI)

        cv2.waitKey(750)

    # Todo: figure out how image pixels translate to 3d position
    return relativeX, relativeY, relativeZ


def getMOTCenter(img):
    """
    Calculates the MOT center based on the innermost contour.

    Contour boundaries are created dynamically based on image intensity.
    Under normal circumstances will create 6 contours evenly spaced between
    intensities: [image max intensity * 0.9, image max intensity * 1/e^2].

    Returns centroid position of innermost contour in camera pixels
    and a copy of the ogImage with contours drawn on it

    :param img: (2D np.array) original roi of raw image of MOT

    :return: x position of mot, y position of mot, image of mot with
             contours drawn on them
    """

    imgCopy = img.copy()

    # Dynamic contour band creation
    upperBound = imgCopy.max() * 0.9
    lowerBound = imgCopy.max() * 0.1353 # 1/e^2 convention
    numContours = 6
    contourBounds = np.linspace(lowerBound, upperBound, numContours)
    # print(contourBounds)

    cnts = []
    for sig in contourBounds:
        contour = getContours(sig, imgCopy, draw=False)
        if contour:  # If the contour list is not empty, add it
            cnts.append(contour)


    for c in cnts[-1]:

        if len(c) <= 2:

            # opencv does not recognize a 2 point contour as closed, so it can't
            # calculate the image moment. In the case of 2 points in the
            # contour just calculate it manually
            # https://en.wikipedia.org/wiki/Image_moment
            m = {
                "m00": len(c),
                "m10": np.sum(c[:, :, 0]),  # sum of pixel x values
                "m01": np.sum(c[:, :, 1]),  # sum of pixel y values
            }
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])

        else:
            # When there are more than 2 points, opencv can recognize
            # the contour is closed and calculate the moments based on
            # cv2.moments
            m = cv2.moments(c)

            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])

    cv2.circle(imgCopy, (cX, cY), 1, 0, -1)
    return cX, cY, imgCopy


def getContours(limit, img, draw=False):
    """
    Given a grayscale threshold, finds contours of the image as a cv2.contours list

    :param limit: (int [0, 255]) Limit for threshold
    :param img: (2D np.array) Source image
    :param draw: (Bool) whether or not to draw the contours on the source image

    :return: list of cv2 contours for the given threshold
    """
    smoothedImg = cv2.GaussianBlur(img, (1, 1), cv2.BORDER_DEFAULT)

    ret, mask = cv2.threshold(smoothedImg, limit, 255, cv2.THRESH_BINARY)

    # Updated versions of findContours removed one of the return parameters
    contours, hierarchy = cv2.findContours(mask, 1, 2)

    if draw:
        # cv2.drawContours(img, contours, -1, 0, 1)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mask', 400, 400)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)

    return contours


def getFiberCenter(img):
    """
    Finds the position of the fiber tip given an image with the Picameras

    :param img: (2D np.array) original roi of raw image of MOT
    :return: (int tuple) x, y position of the fiber tip in camera pixels
    """

    # Threshold image to just show fiber
    ret, imgThresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)

    # Find the x of the fiber based on the largest column intensity
    colSums = np.sum(imgThresh, axis=0)
    x_fiberTip = np.argmax(colSums)

    # Find the pixel column that represents the fiber
    fiberCol = img[:, x_fiberTip].astype('int32')
    startRow = 20
    fiberColDiff = np.diff(fiberCol[startRow:30])

    # Find the tip of the fiber based on the largest difference in intensity
    y_fiberTip = np.argmax(fiberColDiff) + startRow

    return x_fiberTip, y_fiberTip


if __name__ == "__main__":
    ###########################################################################
    # May 5 Alignment image test
    ###########################################################################

    # ~ filePath = r"./saved_images/May 5 0514 us shutter speed images/20210505_162330.jpg"
    # ~ filePath = r"./saved_images/May 5 0514 us shutter speed images/20210505_162332.jpg"

    # ~ filePath = r"./saved_images/20210505_164652.jpg"
    filePath = r"./saved_images/20210505_164654.jpg"
    # ~ filePath = r"./saved_images/20210505_164656.jpg"
    # ~ filePath = r"./saved_images/20210505_164701.jpg"
    # ~ filePath = r"./saved_images/20210505_164703.jpg"


    motImg = cv2.imread(filePath, 0)

    w, h = np.shape(motImg)

    cam0Img = motImg[:h//2, :]  # cropped picture of mot for cam 0
    cam1Img = motImg[h//2:, :]  # cropped picture of mot for cam 0

    x, y, z = getFiberMOTDistance(cam0Img, cam1Img, debug=True)
    print(x, y)

