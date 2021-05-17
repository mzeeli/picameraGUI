"""
Main script for raspberry pi camera GUI.

Creates tkinter GUI application to access mot information with picameras.
Provides functionality to measure alignment, #atoms, temperature.

Last Updated: Summer Term, 2021
Author: Michael Li
"""
from motCamera import MOTCamera
from datetime import datetime
from PIL import ImageTk, Image
from tkinter import filedialog as fd

import tkinter as tk
import tkinter.font as tkFont
import numpy as np
import RPi.GPIO as GPIO

import threading
import cv2
import time
import motTemperature
import motAlignment
import numAtoms
import utils
import logTable


class PiCameraGUI(tk.Frame):
    """
    Main GUI class for dual picamera system functionalities

    Uses tkinter to create a gui for the dual picamera system. The gui consists of
    5 main views ["Alignment", "Analysis", "Camera VIew", "3D View" and "Log View"].
    Each of these 5 views are handled independently by a show{viewName} method and
    chosen by the navigation toolbar created by the createNavigationBtn method

    For an example on how to run the PiCameraGUI see the the if __name__ == __main__
    section of the script

    Fields:
        cam0: MOTCamera class to control cam0 on the raspberry pi
        cam1: MOTCamera class to control cam1 on the raspberry pi
        camOn: Boolean so that the gui can be run even without the cameras attached
        currWin: str to keep track of which of the 5 views we are currently on
        debug: Boolean to enable debugging mode for the GUI
        defaultFont: str of font name to use in the GUI
        log: str array to keep track of most previous actions
        mHeight: int of main window height
        mWidth: int of main window width
        mainDisplay: tk.Frame that shows the current view
        master: tk.Frame that contains the mainDisplay frame and navigation frame
        motx: float x position of mot in um
        moty: float y position of mot in um
        motz: float z position of mot in um
        numAtomsAbs: float #atoms measured by absorption imaging
        numAtomsFlo: float #atoms measured by fluorescent imaging
        output: filepath to save all data (images, txt files)
        temperature: float temp measured by absorption imaging
    """
    def __init__(self, master, output=r"./saved_images", debug=False,
                 camOn=True):
        """
        Constructor

        :param master: (tk.Tk) Main tkinter frame to put everything on
        :param output: (str: filepath) Directory to save images
        :param debug: (bool) Enter debug mode, by default is disabled
        :param camOn: (bool) Turn on cameras, by default enabled. Useful for
                      debugging non-camera related features and code development
                      outside of raspberry pi
        """
        self.master = master
        self.output = output
        self.debug = debug
        self.camOn = camOn

        self.mWidth = 805  # Set main window width
        self.mHeight = 595  # Set main window height
        self.defaultFont = 'Courier'  # Default font style
        self.log = []

        self.motx = -1  # x position of mot relative to fiber
        self.moty = -1  # y position of mot relative to fiber
        self.motz = -1  # z position of mot relative to fiber

        self.temperature = "TBD"  # Temperature
        self.numAtomsAbs = "TBD"  # Number of atoms calculated by absorption
        self.numAtomsFlo = "TBD"  # Number of atoms calculated by fluorescence

        self.currWin = ""  # keeps track of which view we are currently on

        if camOn:
            # Pi compute module assigns cam0 port as numerical value 1
            self.cam0 = MOTCamera(1, grayscale=True)
            self.cam1 = MOTCamera(0, grayscale=True)

        tk.Frame.__init__(self, master)

        # Create main panel and show
        self.mainDisplay = tk.Frame(master=self.master,
                                    height=self.mHeight, width=self.mWidth,
                                    highlightbackground="black",
                                    highlightthicknes=1)
        self.mainDisplay.pack()

        # Default starting window
        self.showAlignmentWin()

        # Add the navigation buttons at the bottom
        self.createNavigationBtn()

    def createNavigationBtn(self):
        """
        Creates and displays the bottom navigation buttons below main window
        """
        btnHeight = 70
        btnWidth = 161

        btnFonts = tkFont.Font(family=self.defaultFont, size=15)

        navigationFrame = tk.Frame(master=self.master,
                                   height=btnHeight + 2, width=self.mWidth,
                                   highlightbackground="black",
                                   highlightthicknes=1)

        # Create Buttons #
        alignmentBtn = tk.Button(navigationFrame, text='Alignment',
                                 font=btnFonts, command=self.showAlignmentWin)
        alignmentBtn.place(x=0, y=0, height=btnHeight, width=btnWidth)

        analysisBtn = tk.Button(navigationFrame, text='Analysis',
                                font=btnFonts, command=self.showAnalysisWin)
        analysisBtn.place(x=btnWidth, y=0, height=btnHeight, width=btnWidth)

        cameraBtn = tk.Button(navigationFrame, text='Camera View',
                              font=btnFonts, command=self.showCameraWin)
        cameraBtn.place(x=btnWidth * 2, y=0, height=btnHeight, width=btnWidth)

        viewBtn = tk.Button(navigationFrame, text='3D View', font=btnFonts,
                            command=self.show3DWin)
        viewBtn.place(x=btnWidth * 3, y=0, height=btnHeight, width=btnWidth)

        logBtn = tk.Button(navigationFrame, text='Log', font=btnFonts,
                           command=self.showLogWin)
        logBtn.place(x=btnWidth * 4, y=0, height=btnHeight, width=btnWidth)

        navigationFrame.pack()

    def showAlignmentWin(self):
        """
        Creates widgets associated with the alignment window.

        This window is used to see how the MOT is aligned compared to the fiber
        """
        self.clearMainDisplay()
        self.currWin = "alignment"  # keep track of current window

        # Create right side information panel #
        coordinatesFrame = tk.Frame(self.mainDisplay, height=570, width=225,
                                    highlightbackground="black",
                                    highlightthicknes=1)
        coordinatesFrame.place(x=570, y=11)

        motLblFont = tkFont.Font(family=self.defaultFont, size=15)
        tk.Label(coordinatesFrame, text='MOT Position (um)', font=motLblFont) \
            .place(relx=0.5, rely=0.05, anchor='center')

        # Display Data
        dataFont = tkFont.Font(family=self.defaultFont, size=20)
        btnRelYStart = 0.18
        btnRelDist = 0.2  # increments in rely between parameters
        xLbl = tk.Label(coordinatesFrame, text=f'x\n{self.motx}', font=dataFont)
        xLbl.place(relx=0.5, rely=btnRelYStart, anchor='center')
        yLbl = tk.Label(coordinatesFrame, text=f'y\n{self.moty}', font=dataFont)
        yLbl.place(relx=0.5, rely=btnRelYStart + btnRelDist, anchor='center')
        zLbl = tk.Label(coordinatesFrame, text=f'z\n{self.motz}', font=dataFont)
        zLbl.place(relx=0.5, rely=btnRelYStart + btnRelDist * 2, anchor='center')
        nLbl = tk.Label(coordinatesFrame, text=f'#Atoms\n{111}', font=dataFont)
        nLbl.place(relx=0.5, rely=btnRelYStart + btnRelDist * 3, anchor='center')

        # Draw MOT on grid #
        gHeight = 570  # Grid height
        gWidth = 550  # Grid width
        alignmentGrid = tk.Canvas(self.mainDisplay, bg="gray60",
                                  height=gHeight, width=gWidth)
        alignmentGrid.place(x=15, y=10)

        # Draw axis lines
        alignmentGrid.create_line(0, gHeight / 2, gWidth, gHeight / 2)
        alignmentGrid.create_line(gWidth / 2, 0, gWidth / 2, gHeight)

        # Get position and number of atoms data
        # Todo get real atom cloud position
        posThread = threading.Thread(target=self.getMotFiberPosition,
                                     args=(xLbl, yLbl, zLbl, nLbl, alignmentGrid))

        btnFont = tkFont.Font(family=self.defaultFont, size=15)
        getPosBtn = tk.Button(coordinatesFrame, text='Get Positions',
                              font=btnFont,
                              command=lambda: posThread.start())
        getPosBtn.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

    def showAnalysisWin(self):
        """
        Creates widgets associated with the Analysis window.

        This window is used to calculate temperature and #Atoms by absorption,
        it will not reflect real-time data like the other windows. The data
        displayed here are fed by selecting MOT images and their background
        photos
        """
        self.clearMainDisplay()
        self.currWin = "analysis"  # keep track of current window

        # Set fonts to be used in labels
        lblFont = tkFont.Font(family=self.defaultFont, size=25)
        dataFont = tkFont.Font(family=self.defaultFont, size=30)

        # Todo get real temperature and atom count
        # Todo create pop up window that allows user to select a bunch of images

        # Display temperature data
        tk.Label(self.mainDisplay, text=f'Temperature (K)', font=lblFont) \
            .place(relx=0.33, rely=0.25, anchor='center')
        tempLbl = tk.Label(self.mainDisplay, text=self.temperature,
                           font=dataFont)
        tempLbl.place(relx=0.33, rely=0.35, anchor='center')

        # Display atom count data
        tk.Label(self.mainDisplay, text=f'#Atoms Abs.', font=lblFont) \
            .place(relx=0.33, rely=0.65, anchor='center')
        numAtomLbl = tk.Label(self.mainDisplay, text=self.numAtomsAbs,
                              font=dataFont)
        numAtomLbl.place(relx=0.33, rely=0.75, anchor='center')

        # button for temperature calculations
        getTempBtn = tk.Button(self.mainDisplay, height=2, width=30,
                               text="Calculate Temperature",
                               command=lambda: self.getTemperature(tempLbl))
        getTempBtn.place(relx=0.8, rely=0.3, anchor='center')

        # button for absorption based #atoms calculations
        getTempBtn = tk.Button(self.mainDisplay, height=2, width=30,
                               text="Get #Atoms by Absorption",
                               command=lambda: self.getNumAtomsAbs(numAtomLbl))
        getTempBtn.place(relx=0.8, rely=0.7, anchor='center')

    def showCameraWin(self):
        """
        Creates widgets associated with the Camera window.

        This window is used to snap and save images from the camera and
        create the video view popout window
        """
        self.clearMainDisplay()
        self.currWin = "camera"  # keep track of current window

        # Wait for camera resource to close on other threads, if BNC 
        # checking thread is also running it should be != 2, as you have 
        # the main gui thread and the BNC thread
        while threading.active_count() != 1:
            time.sleep(0.2)

        ## Main camera Displays ##
        camDispHeight = self.cam1.defaultResY;
        camDispWidth = self.cam1.defaultResX;

        cam0Lbl = tk.Label(self.mainDisplay, bd=1, relief='solid',
                           width=camDispWidth, height=camDispHeight)
        cam0Lbl.place(x=40, y=25)
        # Snap an image upon opening camera view window and display it on cam0 label
        threading.Thread(target=lambda: self.cam0.showImgOnLbl(cam0Lbl)).start()

        cam1Lbl = tk.Label(self.mainDisplay, bd=1, relief='solid',
                           height=camDispHeight, width=camDispWidth)
        cam1Lbl.place(x=40, y=300)
        # Snap an image upon opening camera view window and display it on cam1 label
        threading.Thread(target=lambda: self.cam1.showImgOnLbl(cam1Lbl)).start()

        ## Camera Labels ##
        btnRelx = 0.868
        distY = 60
        lblOffset = 0.08
        btnH = 60
        btnW = 65

        camFont = tkFont.Font(family=self.defaultFont, size=13)
        tk.Label(self.mainDisplay, text='cam0', font=camFont, bg='gray83') \
            .place(x=41, y=26)
        tk.Label(self.mainDisplay, text='cam1', font=camFont, bg='gray83') \
            .place(x=41, y=301)

        ## Shutter Speed Tuning ##
        # Todo add text box for direct ss change
        # Create shutter speed frame
        ssFrame = tk.Frame(master=self.mainDisplay,
                           height=160, width=190,
                           highlightbackground="black",
                           highlightthicknes=1)
        # Shutter Section Label
        tk.Label(ssFrame, text='Shutter Speed (us)', font=camFont) \
            .place(relx=0.5, rely=0.1, anchor='center')

        # Value Display
        currShutterSpeed = f'Current SS: {self.cam1.shutter_speed}'
        ssLbl = tk.Label(ssFrame, text=currShutterSpeed, font=camFont)
        ssLbl.place(relx=0.5, rely=0.3, anchor='center')

        # Calibration Button
        calibrateBtn = tk.Button(ssFrame, relief=tk.GROOVE,
                                 text="Calibrate",
                                 command=lambda:
                                 self.calibrateCameraShutter(ssLbl, ssScale))
        calibrateBtn.place(relx=0.5, rely=0.58, anchor='center')

        # Scale
        ssScale = tk.Scale(ssFrame, from_=0, to=3000,
                           orient=tk.HORIZONTAL, length=180,
                           command=lambda val, lbl=ssLbl:
                           self.setShutterSpeed(lbl, val))

        ssScale.set(self.cam1.shutter_speed)
        ssScale.place(relx=0.5, rely=0.82, anchor='center')

        ssFrame.place(relx=btnRelx, rely=0.18, anchor='center')

        ## Video Button for cam 0 ##
        vidImgPath = r'./assets/vid_0.png'
        vidImg = utils.resizeImage(vidImgPath, btnH, btnW)
        vidBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, command=lambda:
        threading.Thread(target=self.cam0.showVid).start())
        vidBtn.image = vidImg
        vidBtn.configure(image=vidImg)
        vidBtn.place(relx=btnRelx, rely=0.42, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Show Vid 0')\
        # ~ .place(relx=btnRelx, rely=0.2+lblOffset, anchor='center')

        ## Snap Image Button ##
        snapImgPath = r'./assets/snap.png'
        snapImg = utils.resizeImage(snapImgPath, btnH, btnW)
        snapBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE,
                            command=lambda: self.snapImages(cam0Lbl, cam1Lbl))
        snapBtn.image = snapImg
        snapBtn.configure(image=snapImg)
        snapBtn.place(relx=btnRelx, rely=0.57, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Snap Pictures')\
        # ~ .place(relx=btnRelx, rely=0.4+lblOffset, anchor='center')

        ## Save Button ##
        saveImgPath = r'./assets/save.png'
        saveImg = utils.resizeImage(saveImgPath, btnH, btnW)
        saveBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE,
                            command=self.saveImage)
        saveBtn.image = saveImg
        saveBtn.configure(image=saveImg)
        saveBtn.place(relx=btnRelx, rely=0.72, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Save Images')\
        # ~ .place(relx=btnRelx, rely=0.6+lblOffset, anchor='center')

        ## Video Button for cam1 ##
        vidImgPath = r'./assets/vid_1.png'
        vidImg = utils.resizeImage(vidImgPath, btnH, btnW)
        vidBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, command=lambda:
        threading.Thread(target=self.cam1.showVid).start())
        vidBtn.image = vidImg
        vidBtn.configure(image=vidImg)
        vidBtn.place(relx=btnRelx, rely=0.87, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Show Vid 1')\
        # ~ .place(relx=btnRelx, rely=0.8+lblOffset, anchor='center')

    def show3DWin(self):
        """
        Creates widgets associated with the 3D window.

        This window is used to show a 3D view of the MOT based on two-view
        geometry. Todo
        """
        self.clearMainDisplay()
        self.currWin = "3D"  # keep track of current window

        tk.Label(self.mainDisplay, text="Coming soon (maybe)") \
            .place(relx=0.5, rely=0.5, anchor='center')

    def showLogWin(self):
        """
        Creates widgets associated with the Log window.

        This window is used to show main commands used in the past and
        system responses.
        """
        self.clearMainDisplay()
        self.currWin = "log"  # keep track of current window

        logTable.Table(self.mainDisplay, self.log)

    def getMotFiberPosition(self, xLbl, yLbl, zLbl, nLbl, canvas):
        """
        Displays the relative position between the mot and fiber

        Calculates position from mot to fiber using motAlignment.getFiberMOTDistance
        and dispalys them on the alignment view's numerical labels and cartesian grid

        The camera pixel to um conversion is hardcoded to 65 um/pixel. Because the
        130 um fiber appears to be ~2 pixels on the camera

        :param xLbl: (tk.Label) Label to display x position of MOT
        :param yLbl: (tk.Label) Label to display y position of MOT
        :param zLbl: (tk.Label) Label to display z position of MOT
        :param nLbl: (tk.Label) Label to display number of atoms
        :param canvas: (tk.Canvas) Canvas used to draw cartesian coordinates of MOT

        :param imgXSize: (int) image width, should match labels size in camera view
        :param imgYSize: (int) image height, should match labels size in camera view

        """
        # calculations
        print("Started mot-fiber distance calculations")

        gHeight = 570  # Grid height
        gWidth = 550  # Grid width
        motRadius = 15  # Todo dynamic radius?
        zoom = 12  # arbituary zoom factor so that mot appears more dynamic on grid
        px2Micron = 65  # 130 um HCF appears to be ~2 pixels on the camera

        # Create initial mot on canvas but make it off screen
        mot = canvas.create_oval(gWidth / 2 + 1000 - motRadius,
                                 gHeight / 2 - 1000 - motRadius,
                                 gWidth / 2 + 1000 + motRadius,
                                 gHeight / 2 - 1000 + motRadius,
                                 fill='slate gray')
                                 
        # while we are on the alignment window, continue performing alignment
        while self.currWin == "alignment":
            try:
                startTime = time.time()
                # Capture image to cam0.img
                # start image capture on separate thread to take image faster
                cam0Thread = threading.Thread(target=self.cam0.capImgCV2,
                                              args=(self.cam1.defaultResX, 
                                              self.cam1.defaultResY))
                cam0Thread.start()

                # Capture image to cam1.img
                self.cam1.capImgCV2(self.cam1.defaultResX, 
                                    self.cam1.defaultResY)

                cam0Thread.join()  # wait for both images to be captured

                x, y, z = motAlignment.getFiberMOTDistance(self.cam0.img,
                                                           self.cam1.img,
                                                           debug=True)

                # Check again if currently on alignment view and edit labels
                if self.currWin == "alignment":

                    # Edit numerical labels
                    xLbl.configure(text=f"x\n{x * px2Micron}")
                    yLbl.configure(text=f"y\n{y * px2Micron}")
                    zLbl.configure(text=f"z\n{z * px2Micron}")

                    # Edit MOT position on 2d grid
                    canvas.coords(mot,
                                  gWidth / 2 + x * zoom - motRadius,
                                  gHeight / 2 - y * zoom - motRadius,
                                  gWidth / 2 + x * zoom + motRadius,
                                  gHeight / 2 - y * zoom + motRadius)

                    # sleep between frames to reduce load on memory
                    # otherwise cameras may crash
                    time.sleep(0.08)
                    print("total time:", time.time() - startTime)

                else:
                    cv2.destroyAllWindows()
                    break

            except Exception as e:
                print("-----Couldn't find mot-----")
                print("Error", e)

        print("Exiting mot-fiber distance calc")

    def getTemperature(self, tempLbl):
        """
        Calculates temperature based on local absorption images

        Outline:
        1. Select images to use for temperature calculations (suggest > 5)
        2. Select background image of just probe
        3. Sends filepaths to getTempFromImgList calculates temprature
        4. Assigns temperature to self.temp and updates the display label

        :param tempLbl: (tk.Label) Lable to display temperature data on

        :return: None
        """

        intialDir = r"./saved_Images"

        # fd brings up windows explorer to find image files
        fileNames = fd.askopenfilenames(initialdir=intialDir,
                                        title="Select >2 MOT image files")

        bgImgPath = fd.askopenfilename(initialdir=intialDir,
                                       title="Select background image")
        time.sleep(0.2)  # Give time for GUI to update

        # Perform calculations if user selected files
        if len(fileNames) > 2 and bgImgPath:
            print("Running Temperature Calculations")
            tempLbl.configure(text="Calculating")
            self.mainDisplay.update()

            # Get Temperature from selected images
            T = motTemperature.getTempFromImgList(fileNames, bgImgPath)

            # Convert T to string and reduce significant figures
            self.temperature = "{0:.4e}".format(T)

            self.logAction(f"Temperature measured: {self.temperature}")

            # Update label temperature display
            tempLbl.configure(text=self.temperature)

        # Exit funciton if user decided to cancel temp calculations
        else:
            if len(fileNames) <= 2:
                print("Please select more than 2 images for more accurate "
                      "temperature calculation")
                print("Cancelling temperature calculation")

            self.logAction(
                "Cancelled temperature calculation. Incorrect file Selection")
            self.logAction("Please select > 2 MOT images for temp calculation")
            tempLbl.configure(text="Invalid Selection")
            return

    def getNumAtomsAbs(self, numAtomsLbl):
        """
        Calculates #atoms based on local absorption images

        Outline:
        1. Select MOT image (suggest select one of the earlier images)
        2. Select background image of just probe
        3. Select dark bg image
        4. Sends filepaths to numAtomsAbs and calculates #atoms
        5. Assigns #atoms to self.numAtomsAbs and updates the display label

        :param numAtomsLbl: (tkinter.Label) lable to display #atom data on
        :return:
        """
        intialDir = r"./"

        # fd brings up windows explorer to find image files
        motImgPath = fd.askopenfilename(initialdir=intialDir,
                                        title="Select MOT image")

        probeImgPath = fd.askopenfilename(initialdir=intialDir,
                                          title="Select background image")

        bgImgPath = fd.askopenfilename(initialdir=intialDir,
                                       title="Select BG image")
        time.sleep(0.2)  # Give time for GUI to update
        # Only perform calculations if user input paths
        if motImgPath and probeImgPath and bgImgPath:
            numAtomsLbl.configure(text="Calculating")
            self.mainDisplay.update()

            self.numAtomsAbs = numAtoms.numAtomsAbs([motImgPath],
                                                    [probeImgPath],
                                                    [bgImgPath])

            self.numAtomsAbs = "{0:.4e}".format(self.numAtomsAbs)

            self.logAction("#Atom by abssorption calculated: "
                           f"{self.numAtomsAbs}")

            numAtomsLbl.configure(text=self.numAtomsAbs)

        else:
            self.logAction("Cancelled #Atom calc. Incorrect file selection")
            numAtomsLbl.configure(text="Invalid Selection")
            return

    def setShutterSpeed(self, ssLbl, ss):
        """
        Get shutter speed from slider then set them to cameras and update label

        :param ssLbl: (tk.Label) Label to display updated shutter speed
        :param ss: (int) Shutter speed from slider [us]
        :return:
        """
        self.cam0.shutter_speed = int(ss)
        self.cam1.shutter_speed = int(ss)

        ssLbl.configure(text=f'Current SS: {ss}')

    def calibrateCameraShutter(self, ssLbl, ssScale):
        """
        Calibrate shutter speed such that no pixel is saturated

        Using cam1, finds the max shutter speed where no pixel is saturated
        and assigns the same shutter speed value to cam0. Finally updates both the
        shutter speed scale and label to show the new calibrated value.
        """

        ssLbl.configure(text="Calibrating")
        self.mainDisplay.update()  # Refresh screen to say calibrating

        # Set upper bound for shutter speed
        self.cam1.shutter_speed = 20000
        shutterSpeed = self.cam1.calibrateShutterSpeed()
        self.cam0.shutter_speed = shutterSpeed

        # Update widgets
        ssLbl.configure(text=f'Current SS: {shutterSpeed}')
        ssScale.set(shutterSpeed)

        self.logAction(f"Shutter speed calibrated: {shutterSpeed} us")

    def snapImages(self, cam0Lbl, cam1Lbl):
        """
        Snaps new image from both cameras and shows them on the designated labels

        :param cam0Lbl: (tk.Label) Label to display new image from cam0
        :param cam1Lbl: (tk.Label) Label to display new image from cam1
        """

        self.logAction("Snapped images")
        threading.Thread(target=lambda: self.cam0.showImgOnLbl(cam0Lbl)).start()
        threading.Thread(target=lambda: self.cam1.showImgOnLbl(cam1Lbl)).start()

    def saveImage(self, iden=""):
        """
        Saves the last images snapped by the cameras.

        Grab both images as np.arrays and stack them veritcally. Finally saving them
        by their timestamp in the self.output folder.
        
        :param iden: (str) Optional unique identifier for file name.
        """

        # Create timestamp and file path
        timestamp = utils.getTimestamp()

        if iden:
            savePath = f"{self.output}/{iden}_{timestamp}.jpg"

        else:
            savePath = f"{self.output}/{timestamp}.jpg"

        # Stack cam0 and cam1 images vertically
        combinedImg = np.concatenate((self.cam0.img, self.cam1.img), axis=0)

        # Save combined image
        cv2.imwrite(savePath, combinedImg)
        self.logAction(f"Saved image to {savePath}")

    def clearMainDisplay(self):
        """
        Clears all widgets on the display to make room on the new window.

        Needs to be called at the start of every new window call
        """

        for widget in self.mainDisplay.winfo_children():
            widget.destroy()

    def logAction(self, msg):
        """
        Adds new action to log list

        If the list exceeds 21 (length of the display) then it removes the oldest
        action and moves everything up by 1 index. This way the log view appears as
        the newest actions are at the top and the oldest ones get popped out.

        :param msg: (str) Message of last action to add to log list
        """
        # If display log becomes full, move everything up 1 to make space
        timestamp = datetime.now().strftime("%I:%M:%S %p")

        if len(self.log) >= 21:
            for i in range(1, 20):
                self.log[i] = self.log[i + 1]

            self.log[20] = (timestamp, msg)

        # If display log not full just add to end
        else:
            self.log.append((timestamp, msg))

    def onWinClose(self):
        """
        Callback for when the gui is closed.

        Need to ensure the camera resources are released
        """
        print("Exiting Application")

        # Release camera resources
        if self.camOn:
            self.cam0.close()
            self.cam1.close()

        self.master.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    window.title('PiCamera')

    # Initiate GUI
    gui = PiCameraGUI(window, debug=False, camOn=True)

    # Create upon program exit
    window.protocol("WM_DELETE_WINDOW", gui.onWinClose)

    gui.pack()

    window.mainloop()
