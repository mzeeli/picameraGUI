import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def create3DView(debug=False):
    """
    """

    # Top View #
    topBack = cv2.imread(r"./saved_images/20210202_213251.jpg", 0)  # Background photo in grayscale
    topTarget = cv2.imread(r"./saved_images/20210202_213301.jpg", 0)  # Photo of Cs in grayscale
    topView = subtractBackground(topBack, topTarget)

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
    sideBack = cv2.imread(r"./saved_images/20210202_213648.jpg", 0)  # Background photo in grayscale
    sideTarget = cv2.imread(r"./saved_images/20210202_213631.jpg", 0)  # Photo of Cs in grayscale
    sideView = subtractBackground(sideBack, sideTarget)

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
        cv2.resizeWindow('Top No Background', int(w * scaling), int(h * scaling))
        cv2.resizeWindow('Side No Background', int(w * scaling), int(h * scaling))
        cv2.resizeWindow('Top Binary', int(w * scaling), int(h * scaling))
        cv2.resizeWindow('Side Binary', int(w * scaling), int(h * scaling))

        cv2.imshow("Top No Background", topTarget)
        cv2.imshow("Side No Background", sideTarget)
        cv2.imshow("Top Binary", topMask)
        cv2.imshow("Side Binary", sideMask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def subtractBackground(imgBack, imgTarget):
    """
    imgBack (np.array: uint8) = Array image of background
    imgTarget (np.array: uint8) = Array image of target
    """
    noBackground = imgTarget.astype(int) - imgBack.astype(int)
    noBackground[noBackground < 0] = 0  # restrain uint8 values
    noBackground = noBackground.astype(np.uint8)

    return noBackground

def createMask(img):
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    ret, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def matchPoints(mask1, mask2):
    """
    Intakes two binary images of [0, 255] and creates an array of 3D points from them

    :param mask1: Binary image mask of first view
    :param mask2:  Binary image mask of second view

    :return points: nx3 np.array of [x,y,z] points to plot in matplotlib
    """
    # cv2.imshow("m1", mask2)

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

def getImageCenter(img):
    """

    :param img: image of MOT

    :return: relative distance to fiber in pixels
    """

    cnts = []
    for sig in [40, 80, 120, 160, 200, 240]:
        contour = getContours(sig, img)
        cnts.append(contour)

    # Get center of mass based on outermost contour
    for c in cnts[0]:
        m = cv2.moments(c)
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])

        cv2.circle(img, (cX, cY), 3, 0, -1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    return 0

def getContours(limit, img):
    smoothedImg = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    ret, thresh = cv2.threshold(smoothedImg, limit, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.drawContours(img, contours, -1, 0, 1)
    return contours


if __name__ == "__main__":
    # create3DView(debug=True)
    imgPath = r"./saved_images/mot image.png"
    image = cv2.imread(imgPath, 0)
    dist = getImageCenter(image)
    print(dist)