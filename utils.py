"""
Utility funcitons for the picamera gui

Last Updated: Summer Term, 2021
Author: Michael Li
"""

from datetime import datetime
from PIL import ImageTk, Image


def getTimestamp():
    """
    :return: Timestamp in form of YYYYmmDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resizeImage(imgPath, h, w):
    """
    Resizes images to use on tkinter

    Maps image to the dimensions of h and w and then returns it as an
    ImageTk.PhotoImage object to use in tkinter widgets
    :param imgPath: file path to image
    :param h: desired image height in pixels
    :param w: desired image width in pixels
    :return: image to use in tkinter widgets
    """

    img = Image.open(imgPath)
    img = img.resize((w, h), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    return img
