"""
Calculate mot position given two pi camera images

The Pi cameras are expected to be placed at roughly 90 degrees from each 
other

Last Updated: Summer Term, 2021
Author: Michael Li
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROIs for when both cameras are in front
# Determined these values just based on trial and error
cam0xROI = 245
cam0yROI = 130
cam1xROI = 205
cam1yROI = 145
lengthROI = 80

def create3DView(debug=False):
    """
    Test function to create a 3D scene based on two-view geometry
    """

    # Top View #
    # Background photo in grayscale
    topBack = cv2.imread(r"./saved_images/20210202_213251.jpg", 0)
    # Photo of Cs in grayscale
    topTarget = cv2.imread(r"./saved_images/20210202_213301.jpg", 0)
    topView = cv2.subtract(topBack, topTarget)

    topMask = createMask(topView)
    topMask = cv2.rotate(topMask, cv2.ROTATE_90_CLOCKWISE)
    topTarget = cv2.rotate(topTarget, cv2.ROTATE_90_CLOCKWISE)
    topMask = topMask[:345, :]
    topTarget = topTarget[:345, :]

    # Get Tighter crop
    xTop, yTop = np.where(topMask == 255)
    yCropMin = min(yTop)
    yCropMax = max(yTop)
    topMask = topMask[:, yCropMin:yCropMax]
    topTarget = topTarget[:, yCropMin:yCropMax]

    # Side View #
    # Background photo in grayscale
    sideBack = cv2.imread(r"./saved_images/20210202_213648.jpg", 0)
    # Photo of Cs in grayscale
    sideTarget = cv2.imread(r"./saved_images/20210202_213631.jpg", 0)
    sideView = cv2.subtract(sideBack, sideTarget)

    # cv2.imshow('target', sideTarget)
    # cv2.imshow('back', sideBack)
    # cv2.imshow('sideView', sideView)
    # cv2.waitKey(0)

    sideMask = createMask(sideView)
    sideMask = sideMask[:, 220:320]
    sideTarget = sideTarget[:, 220:320]

    # Get Tighter crop
    xSide, zSide = np.where(sideMask == 255)
    zCropMin = min(zSide)
    zCropMax = max(zSide)
    sideMask = sideMask[:, zCropMin:zCropMax]
    sideTarget = sideTarget[:, zCropMin:zCropMax]

    # Resize so both images are same size, make side image the same size as top
    h, w = topMask.shape
    sideMask = cv2.resize(sideMask, (w, h))
    sideTarget = cv2.resize(sideTarget, (w, h))
    print(sideMask.shape)
    # create 3D points
    points = matchPoints(topMask, sideMask)
    everyNthPoint = 200
    x = points[::everyNthPoint, 0].astype(int)
    y = points[::everyNthPoint, 1].astype(int)
    z = points[::everyNthPoint, 2].astype(int)

    # Plotting 1#
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plotting 2#
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

    # Plotting 3#
    # fig = plt.figure()
    # fig.patch.set_facecolor('white')
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z)

    # Plotting 4#
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_wireframe(x, y, z)

    plt.show()
    if debug:
        # Show images
        imgTop = cv2.hconcat([topTarget, topMask])
        imgSide = cv2.hconcat([sideTarget, sideMask])
        h, w = imgTop.shape

        cv2.namedWindow('Top No Background', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Side No Background', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Top Binary', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Side Binary', cv2.WINDOW_NORMAL)

        scaling = 2.3
        cv2.resizeWindow('Top No Background', int(w*scaling), int(h*scaling))
        cv2.resizeWindow('Side No Background', int(w*scaling), int(h*scaling))
        cv2.resizeWindow('Top Binary', int(w * scaling), int(h * scaling))
        cv2.resizeWindow('Side Binary', int(w * scaling), int(h * scaling))

        cv2.imshow("Top No Background", topTarget)
        cv2.imshow("Side No Background", sideTarget)
        cv2.imshow("Top Binary", topMask)
        cv2.imshow("Side Binary", sideMask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def createMask(img):
    """
    Creates a mask hard coded to 15 DN for a given image
    :param img: image to create mask for
    :return: Binary mask with max points set to 255
    """
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    ret, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def matchPoints(mask1, mask2):
    """
    Intakes two binary images of [0, 255] and creates an array of 3D points from
    them based on two-view geometry

    :param mask1: Binary image mask of first view
    :param mask2:  Binary image mask of second view

    :return points: nx3 np.array of [x,y,z] points to plot in matplotlib
    """

    mask1Coords = np.array(np.where(mask1 == 255))  # [xTop, yTop]
    mask2Coords = np.array(np.where(mask2 == 255))  # [xSide, zSide]

    points = np.empty(3)
    for index, x in enumerate(mask2Coords[0]):
        matchedIndex = np.where(mask1Coords[0] == x)[0]
        matchedX = mask1Coords[:, matchedIndex]
        matchedX = np.transpose(matchedX)
        h, _ = matchedX.shape
        y = np.ones((h, 1)) * mask2Coords[1][index]
        points = np.vstack((points, np.hstack((matchedX, y))))

    return points[1:, :]

def getFiberMOTDistance(cam0MotImgRaw, cam1MotImgRaw, debug=False):
    """
    A new get mot center code written for a change in the setup on April 23. Instead of having one camera looking at the
    mot through the coils, both of them now look at them from the front but at a slightly different angle. Now we need
    to adjust the code for it. i.e. since now none of the cameras have light scattering problems we don't need background
    subtraction

    :param cam0MotImgRaw: Raw uncropped grayscale image with mot of camera 0
    :param cam1MotImgRaw: Raw uncropped grayscale image with mot of camera 1
    :param debug: debugging flag
    
    :return: x, y, z of the mot position in camera pixels
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

    Returns centroid position of innermost contour and a copy of the ogImage
    with contours drawn on it

    :param ogImage: original roi of raw image of MOT

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
    Given a grayscale threshold, finds contours of the image and returns them as
    a cv2.contours list. draw == true will draw contours on the original image

    :param limit: (int [0, 255]) Limit for threshold
    :param img: (2D np.array) source image
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

    :param img: original roi of raw image of MOT
    :return: x and y position of the fiber tip
    """
    
    # Threshold image to just show fiber
    ret, imgThresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    # ~ cv2.namedWindow('imgThresh', cv2.WINDOW_NORMAL)
    # ~ cv2.resizeWindow('imgThresh', 400, 400)
    # ~ cv2.imshow("imgThresh", imgThresh)
    # ~ cv2.waitKey(0)
    
    # Find the x of the fiber based on the largest column intensity
    colSums = np.sum(imgThresh, axis=0)
    x = np.argmax(colSums)
    
    # Find the pixel column that represents the fiber
    fiberCol = img[:, x].astype('int32')
    fiberColDiff = np.diff(fiberCol)
    
    # Find the tip of the fiber based on the largest difference in intensity
    # +1 to offset adjust to the diff index
    y = np.argmax(fiberColDiff)
    
    return x, y


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
    
