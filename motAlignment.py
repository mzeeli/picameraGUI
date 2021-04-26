"""
Calculate mot position given images

Last Updated: Winter, 2021
Author: Michael Li
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def getFiberMOTDistance(cam0MotImgRaw, cam1MotImgRaw, debug=True):
    """

    :param cam0MotImgRaw: Raw uncropped grayscale image with mot of camera 0
    :param cam1MotImgRaw: Raw uncropped grayscale image with mot of camera 1
    :param debug:
    :return:
    """

    ################################################################
    # Read background images. These should be constant after we find a good
    # position for the cameras and find a good shutter speed
    ################################################################
    bgImg = cv2.imread(r"./saved_images/Background/background.jpg", 0)

    _, imgH = np.shape(motImg)
    cam0BgImg = bgImg[:imgH//2, :]
    cam1BgImg = bgImg[imgH//2:, :]

    ################################################################
    ## Perform background subtraction
    ################################################################
    cam0MotImg = cv2.subtract(cam0MotImgRaw, cam0BgImg)
    cam1MotImg = cv2.subtract(cam1MotImgRaw, cam1BgImg)

    cam0xROI = 215
    cam0yROI = 105
    cam1xROI = 150
    cam1yROI = 100


    cam0MotROI = cam0MotImg[cam0yROI:cam0yROI + 100, cam0xROI:cam0xROI + 100]
    cam1MotROI = cam1MotImg[cam1yROI:cam1yROI + 100, cam1xROI:cam1xROI + 100]

    cv2.namedWindow('cam0MotROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam0MotROI', 400, 400)
    cv2.imshow("cam0MotROI", cam0MotROI)

    x0, y0, _ = getMOTCenter(cam0MotROI)
    x1, y1, _ = getMOTCenter(cam1MotROI)

    x0 = x0 + cam0xROI  # overall pixel location outside of just ROI
    y0 = y0 + cam0yROI  # overall pixel location outside of just ROI
    x1 = x1 + cam1xROI  # overall pixel location outside of just ROI
    y1 = y1 + cam1yROI  # overall pixel location outside of just ROI

    if debug:
        cv2.circle(cam0MotImg, (x0, y0), 6, 260, 1)
        cv2.circle(cam1MotImg, (x1, y1), 6, 260, 1)
        cv2.circle(cam0MotImgRaw, (x0, y0), 6, 260, 1)
        cv2.circle(cam1MotImgRaw, (x1, y1), 6, 260, 1)
        cv2.circle(cam0BgImg, (x0, y0), 6, 260, 1)
        cv2.circle(cam1BgImg, (x1, y1), 6, 260, 1)

        motCombined = np.vstack((cam0MotImg, cam1MotImg))
        motCombinedRaw = np.vstack((cam0MotImgRaw, cam1MotImgRaw))
        bgCombined = np.vstack((cam0BgImg, cam1BgImg))

        cv2.imshow("MotImg", motCombined)
        cv2.imshow("MotImgRaw", motCombinedRaw)
        cv2.imshow("bgCombined", bgCombined)

        cv2.imwrite("test_motCombined.jpg", motCombined)
        cv2.imwrite("test_motCombinedRaw.jpg", motCombinedRaw)
        cv2.imwrite("test_bgCombined.jpg", bgCombined)

        cv2.namedWindow('cam0MotROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam0MotROI', 400, 400)
        cv2.namedWindow('cam1MotROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam1MotROI', 400, 400)
        cv2.imshow("cam0MotROI", cam0MotROI)
        cv2.imshow("cam1MotROI", cam1MotROI)
        cv2.waitKey(0)

    # Todo: figure out how image pixels translate to 3d position
    motPixelPositions = {
        "x": x0,
        "y": x1,
        "z": (y0+y1)/2,
    }

    return motPixelPositions


def getFiberMOTDistanceCamsFront(cam0MotImgRaw, cam1MotImgRaw, debug=True):
    """
    A new get mot center code written for a change in the setup on April 23. Instead of having one camera looking at the
    mot through the coils, both of them now look at them from the front, but at a slightly different angle. Now we need
    to adjust the code for it. i.e. since now none of the cameras have light scattering problem we don't need background
    subtraction

    :param cam0MotImgRaw: Raw uncropped grayscale image with mot of camera 0
    :param cam1MotImgRaw: Raw uncropped grayscale image with mot of camera 1
    :param debug:
    :return:
    """

    ################################################################
    # Get MOT's location
    ################################################################
    cam0xROI = 250
    cam0yROI = 150
    cam1xROI = 210
    cam1yROI = 160


    cam0MotROI = cam0MotImgRaw[cam0yROI:cam0yROI + 60, cam0xROI:cam0xROI + 60]
    cam1MotROI = cam1MotImgRaw[cam1yROI:cam1yROI + 60, cam1xROI:cam1xROI + 60]

    x0, y0, _ = getMOTCenter(cam0MotROI)
    x1, y1, _ = getMOTCenter(cam1MotROI)

    x0 = x0 + cam0xROI  # overall pixel location outside of just ROI
    y0 = y0 + cam0yROI  # overall pixel location outside of just ROI
    x1 = x1 + cam1xROI  # overall pixel location outside of just ROI
    y1 = y1 + cam1yROI  # overall pixel location outside of just ROI


    ################################################################
    # Get fiber's location
    ################################################################
    fiberx0 = getFiberCenter(cam0MotROI)
    fiberx1 = getFiberCenter(cam1MotROI)

    fiberx0 = fiberx0 + cam0xROI
    fiberx1 = fiberx1 + cam1xROI

    relativeX = x0 - fiberx0
    relativeY = x1 - fiberx1

    print(f"relative distances --- x: {relativeX}, y: {relativeY}")

    if debug:
        cv2.circle(cam0MotImgRaw, (x0, y0), 6, 260, 1)
        cv2.circle(cam1MotImgRaw, (x1, y1), 6, 260, 1)
        cv2.line(cam0MotImgRaw, (fiberx0, 0), (fiberx0, 272), (255, 0, 255), 1)
        cv2.line(cam1MotImgRaw, (fiberx1, 0), (fiberx1, 272), (255, 0, 255), 1)

        motCombinedRaw = np.vstack((cam0MotImgRaw, cam1MotImgRaw))

        cv2.imshow("MotImgRaw", motCombinedRaw)

        cv2.imwrite("test_motCombinedRaw.jpg", motCombinedRaw)

        cv2.namedWindow('cam0MotROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam0MotROI', 400, 400)
        cv2.namedWindow('cam1MotROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam1MotROI', 400, 400)
        cv2.imshow("cam0MotROI", cam0MotROI)
        cv2.imshow("cam1MotROI", cam1MotROI)
        cv2.waitKey(0)

    # Todo: figure out how image pixels translate to 3d position
    motPixelPositions = {
        "x": x0,
        "y": x1,
        "z": (y0+y1)/2,
    }


    return relativeX, relativeY


def getMOTCenter(img):
    """
    Calculates the MOT center based on the innermost contour.
    Contour boundaries are created dynamically based on image intensity.

    Under normal circumstances will create 6 contours evenly spaced between
    intensities: [image max intensity * 0.9, image max intensity * 1/e^2].

    Returns centroid position of innermost contour and a copy of the ogImage
    with contours drawn on it

    :param ogImage: original roi of raw image of MOT

    :return: x position of mot, y position of mot, contours drawn on a copy of
            ogImage
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

    colSums = np.sum(img, axis=0)
    x = np.argmax(colSums)

    return x

def randomTestImages():
    """
    Function to analyze random mot images I found and calculate their distance
    to an arbituary fiber
    :return:
    """
    # imgPath = r"./saved_images/motTestImage.png"
    imgPath = r"./saved_images/motTestImage_pantitaThesis.jpg"
    imgPath = r"./saved_images/mot image.png"
    image = cv2.imread(imgPath, 0)

    x, y, image = getMOTCenter(image)

    # Create arbituary fiber and draw on picture
    h, w = image.shape
    fiberX = int(w / 2)  # Assume fiber is in the middle of the image
    fiberY = h - 50  # Assume fiber is 50 pixels above bottom border
    cv2.rectangle(image, (fiberX - 5, h), (fiberX + 5, fiberY), 190, -1)

    relx = x - fiberX
    rely = fiberY - y
    xPos = f"Rel_x: {relx} px"
    yPos = f"Rel_y: {rely} px"

    cv2.putText(image, xPos, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1, cv2.LINE_AA)
    cv2.putText(image, yPos, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1, cv2.LINE_AA)

    cv2.arrowedLine(image, (fiberX, fiberY), (x, y), 0, 1)

    cv2.imshow("image", image)
    cv2.waitKey(0)

def npqoTestImages():
    # Image 1 #
    imgPath = r"./saved_images/npqo_pics/little_MOT_after_UV.jpg"
    image = cv2.imread(imgPath, 0)
    image = cv2.rotate(image, cv2.ROTATE_180)
    motROI = image[205:245, 340:380]
    x, y, imgResult1 = getMOTCenter(motROI)
    cv2.namedWindow('imgResult1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgResult1', 300, 300)
    cv2.imshow("imgResult1", imgResult1)

    # Image 2 #
    imgPath = r"./saved_images/npqo_pics/3co4master.jpg"
    image = cv2.imread(imgPath, 0)
    image = cv2.rotate(image, cv2.ROTATE_180)
    motROI = image[305:405, 380:480]
    x, y, imgResult2 = getMOTCenter(motROI)
    cv2.namedWindow('imgResult2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgResult2', 300, 300)
    cv2.imshow("imgResult2", imgResult2)

    # Image 3 #
    imgPath = r"./saved_images/npqo_pics/2019-6-07.jpg"
    image = cv2.imread(imgPath, 0)
    image = cv2.rotate(image, cv2.ROTATE_180)
    motROI = image[30:100, 350:420]
    x, y, imgResult3 = getMOTCenter(motROI)
    cv2.namedWindow('imgResult3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgResult3', 300, 300)
    cv2.imshow("imgResult3", imgResult3)

    cv2.waitKey(0)


if __name__ == "__main__":
    # create3DView(debug=True)
    # npqoTestImages()

    # motPath = r"C:\Users\Michael\OneDrive\Co-op 5\NPQO\Pi Camera\Picamera images april 20 - first time mot\mot 1.jpg"
    # BgPath = r"C:\Users\Michael\OneDrive\Co-op 5\NPQO\Pi Camera\Picamera images april 20 - first time mot\background 1.jpg"

    #######################################################
    # for when one of the cameras look through the coils
    #######################################################
    motPath = r"C:\Users\Michael Li\OneDrive\Co-op 5\NPQO\Pi Camera\Picamera images april 20 - first time mot\mot 1.jpg"

    motImg = cv2.imread(motPath, 0)

    w, h = np.shape(motImg)

    cam0Img = motImg[:h//2, :]  # cropped picture of mot for cam 0
    cam1Img = motImg[h//2:, :]  # cropped picture of mot for cam 0

    relPos1 = getFiberMOTDistance(cam0Img, cam1Img)
    relPos1 = [5, -4]
    ###########################################################################
    # April 23 for when both cameras look at it from the front
    ###########################################################################
    motPath = r"C:\Users\Michael Li\OneDrive\Co-op 5\NPQO\Pi Camera\Picamera images april 23 - both cameras in front\20210423_164704.jpg"

    motImg = cv2.imread(motPath, 0)

    w, h = np.shape(motImg)

    cam0Img = motImg[:h//2, :]  # cropped picture of mot for cam 0
    cam1Img = motImg[h//2:, :]  # cropped picture of mot for cam 0

    relPos2 = getFiberMOTDistanceCamsFront(cam0Img, cam1Img)
    print(relPos2)
    angle = 45
    # Apply rotation matrix, assuming 45 degrees turn
    relPos2 = [relPos2[0] * np.cos(angle) - relPos2[1] * np.sin(angle),
               relPos2[0] * np.sin(angle) + relPos2[1] * np.cos(angle)]
    print(relPos2)


    pixel2Micron = 130/1.3

    relPos2 = np.array(relPos2)*pixel2Micron
    relPos1 = np.array(relPos1)*pixel2Micron


    plt.plot(relPos2[0], relPos2[1], ".")
    plt.plot(relPos1[0], relPos1[1], ".")
    plt.legend(["Both cameras in front", "One camera through coil"])
    plt.xlabel("X distance (um)")
    plt.ylabel("Y distance (um)")
    plt.xlim([-7*pixel2Micron,7*pixel2Micron])
    plt.ylim([-7*pixel2Micron,7*pixel2Micron])
    plt.show()


