"""
Main script for raspberry pi camera GUI:
Creates tkinter GUI application to access mot information with picameras.
Provides information on alignment , #atoms, temp

Last Updated: Winter, 2021
Author: Michael Li
"""
from motCamera import MOTCamera
from datetime import datetime
from PIL import ImageTk, Image
from tkinter import filedialog as fd

import tkinter as tk
import tkinter.font as tkFont
import numpy as np
import threading
import cv2

import motTemperature
import motAlignment
import numAtoms

class PiCameraGUI(tk.Frame):
    def __init__(self, master, output=r"./saved_images", debug=False,
                 camOn=True):
        """
        Main GUI class for picamera functionalities

        :param master: Main tkinter frame to put everything on
        :param output: Directory to save images
        :param debug: Enter debug mode, by default is disabled
        :param camOn: Turn on cameras, by default enabled. Useful for
        debugging non-camera related features and code development outside of
        raspberry pi
        """
        self.master = master
        self.output = output
        self.debug = debug
        self.camOn = camOn

        self.mWidth = 800  # Set main window width
        self.mHeight = 600  # Set main window height
        self.defaultFont = 'Courier'  # Default font style
        self.log = []

        self.temperature = "TBD"  # Temperature
        self.numAtomsAbs = "TBD"  # Number of atoms calculated by absorption
        self.numAtomsFlo = "TBD"  # Number of atoms calculated by fluorescence

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
        self.showAnalysisWin()

        # Add the navigation buttons at the bottom
        self.createNavigationBtn()

    def createNavigationBtn(self):
        """
        Creates and displays the bottom navigation buttons and assigns their
        respective window display functions
        """
        btnHeight = 70
        btnWidth = 160

        btnFonts = tkFont.Font(family=self.defaultFont, size=15)

        navigationFrame = tk.Frame(master=self.master,
                                   height=btnHeight+2, width=self.mWidth,
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
        cameraBtn.place(x=btnWidth*2, y=0, height=btnHeight, width=btnWidth)
        
        viewBtn = tk.Button(navigationFrame, text='3D View', font=btnFonts,
                            command=self.show3DWin)
        viewBtn.place(x=btnWidth*3, y=0, height=btnHeight, width=btnWidth)
        
        logBtn = tk.Button(navigationFrame, text='Log', font=btnFonts,
                           command=self.showLogWin)
        logBtn.place(x=btnWidth*4, y=0, height=btnHeight, width=btnWidth)

        navigationFrame.pack()

    def showAlignmentWin(self):
        """
        Creates widgets associated with the alignment window.
        This window is used to see how the MOT is aligned compared to the fiber
        """
        self.clearMainDisplay()

        # Create right side information panel #
        coordinatesFrame = tk.Frame(self.mainDisplay, height=570, width=225,
                                    highlightbackground="black",
                                    highlightthicknes=1)
        coordinatesFrame.place(x=570, y=11)

        motLblFont = tkFont.Font(family=self.defaultFont, size=20)
        tk.Label(coordinatesFrame, text='MOT Position', font=motLblFont)\
            .place(relx=0.5, rely=0.05, anchor='center')

        # Get position and number of atoms data
        # Todo get real atom cloud position

        imgPath = r"./saved_images/mot image.png"
        image = cv2.imread(imgPath, 0)
        x1, z1, _ = motAlignment.getMOTCenter(image)

        positions = np.array([x1, -5, z1])
        numAtoms = 101367

        # Display Data
        dataFont = tkFont.Font(family=self.defaultFont, size=20)
        relYStart = 0.18
        relDist = 0.225  # increments in rely between parameters
        tk.Label(coordinatesFrame, text=f'x\n{positions[0]}', font=dataFont)\
            .place(relx=0.5, rely=relYStart, anchor='center')
        tk.Label(coordinatesFrame, text=f'y\n{positions[1]}', font=dataFont)\
            .place(relx=0.5, rely=relYStart + relDist, anchor='center')
        tk.Label(coordinatesFrame, text=f'z\n{positions[2]}', font=dataFont)\
            .place(relx=0.5, rely=relYStart + relDist * 2, anchor='center')
        tk.Label(coordinatesFrame, text=f'#Atoms\n{numAtoms}', font=dataFont)\
            .place(relx=0.5, rely=relYStart + relDist * 3, anchor='center')

        # Draw MOT on grid #
        gHeight = 570  # Grid height
        gWidth = 550  # Grid width
        alignmentGrid = tk.Canvas(self.mainDisplay, bg="gray60",
                                  height=gHeight, width=gWidth)
        alignmentGrid.place(x=15, y=10)

        # Draw axis lines
        alignmentGrid.create_line(0, gHeight/2, gWidth, gHeight/2)
        alignmentGrid.create_line(gWidth/2, 0, gWidth/2, gHeight)

        # Scale pixel location to real location
        zoom = 1
        positions = positions*zoom
        # Draw MOT
        motRadius = 15  # Todo dynamic radius?
        alignmentGrid.create_oval(gWidth/2+positions[0]-motRadius,
                                  gHeight/2-positions[1]-motRadius,
                                  gWidth/2+positions[0]+motRadius,
                                  gHeight/2-positions[1]+motRadius,
                                  fill='slate gray')

    def showAnalysisWin(self):
        """
        Creates widgets associated with the Analysis window.
        This window is used to calculate temperature and #Atoms by absorption,
        it will not reflect real-time data like the other windows. The data
        displayed here are fed by selecting MOT images and their background
        photos
        """
        self.clearMainDisplay()

        # Set fonts to be used in labels
        lblFont = tkFont.Font(family=self.defaultFont, size=25)
        dataFont = tkFont.Font(family=self.defaultFont, size=30)

        # Todo get real temperature and atom count
        # Todo create pop up window that allows user to select a bunch of images


        # Display temperature data
        tk.Label(self.mainDisplay, text=f'Temperature (K)', font=lblFont)\
            .place(relx=0.35, rely=0.25, anchor='center')
        tempLbl = tk.Label(self.mainDisplay, text=self.temperature,
                           font=dataFont)
        tempLbl.place(relx=0.35, rely=0.35, anchor='center')

        # Display atom count data
        tk.Label(self.mainDisplay, text=f'#Atoms', font=lblFont)\
            .place(relx=0.35, rely=0.65, anchor='center')
        numAtomLbl = tk.Label(self.mainDisplay, text=self.numAtomsAbs,
                              font=dataFont)
        numAtomLbl.place(relx=0.35, rely=0.75, anchor='center')

        # button for temperature calculations
        getTempBtn = tk.Button(self.mainDisplay, height=2, width=30,
                               text="Calculate Temperature",
                               command=lambda : self.getTemperature(tempLbl))
        getTempBtn.place(relx=0.8, rely=0.3, anchor='center')

        # button for absorption based #atoms calculations
        getTempBtn = tk.Button(self.mainDisplay, height=2, width=30,
                               text="Get #Atoms by Absorption",
                               command=lambda : self.getNumAtomsAbs(numAtomLbl))
        getTempBtn.place(relx=0.8, rely=0.7, anchor='center')

    def showCameraWin(self):
        """
        Creates widgets associated with the Camera window.
        This window is used to snap and save images from the camera and
        create the video view popout window
        """
        self.clearMainDisplay()
        
        ## Main camera Displays ##
        camDispHeight = 272;
        camDispWidth = 608;

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
        btnRelx = 0.91
        distY = 60
        lblOffset = 0.08
        btnH = 60
        btnW = 65

        camFont = tkFont.Font(family=self.defaultFont, size=13)
        tk.Label(self.mainDisplay, text='cam0', font=camFont, bg='gray83')\
            .place(x=41, y=26)
        tk.Label(self.mainDisplay, text='cam1', font=camFont, bg='gray83')\
            .place(x=41, y=301)

        ## Calibration Labels
        calibrateBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, 
                                 text="Calibrate",
                                 command=self.calibrateCameraShutter)
        calibrateBtn.place(relx=btnRelx, rely=0.1, anchor='center')
                           
        ## Video Button for cam 0 ##
        vidImgPath = r'./assets/vid_0.png'
        vidImg = resizeImage(vidImgPath, btnH, btnW)
        vidBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, command=lambda:
                           threading.Thread(target=self.cam0.showVid).start())
        vidBtn.image = vidImg
        vidBtn.configure(image=vidImg)
        vidBtn.place(relx=btnRelx, rely=0.2, anchor='center')
        tk.Label(self.mainDisplay, text='Show Vid 0')\
            .place(relx=btnRelx, rely=0.2+lblOffset, anchor='center')

        ## Snap Image Button ##
        snapImgPath = r'./assets/snap.png'
        snapImg = resizeImage(snapImgPath, btnH, btnW)
        snapBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, 
                            command=lambda : self.snapImages(cam0Lbl, cam1Lbl))
        snapBtn.image = snapImg
        snapBtn.configure(image=snapImg)
        snapBtn.place(relx=btnRelx, rely=0.4, anchor='center')
        tk.Label(self.mainDisplay, text='Snap Pictures')\
            .place(relx=btnRelx, rely=0.4+lblOffset, anchor='center')
            
        ## Save Button ##
        saveImgPath = r'./assets/save.png'
        saveImg = resizeImage(saveImgPath, btnH, btnW)
        saveBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE,
                            command=self.saveImage)
        saveBtn.image = saveImg
        saveBtn.configure(image=saveImg)
        saveBtn.place(relx=btnRelx, rely=0.6, anchor='center')
        tk.Label(self.mainDisplay, text='Save Images')\
            .place(relx=btnRelx, rely=0.6+lblOffset, anchor='center')
            
            
        ## Video Button for cam1 ##
        vidImgPath = r'./assets/vid_1.png'
        vidImg = resizeImage(vidImgPath, btnH, btnW)
        vidBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, command=lambda:
                           threading.Thread(target=self.cam1.showVid).start())
        vidBtn.image = vidImg
        vidBtn.configure(image=vidImg)
        vidBtn.place(relx=btnRelx, rely=0.8, anchor='center')
        tk.Label(self.mainDisplay, text='Show Vid 1')\
            .place(relx=btnRelx, rely=0.8+lblOffset, anchor='center')

    def show3DWin(self):
        """
        Creates widgets associated with the 3D window.
        This window is used to show a 3D view of the MOT based on two-view
        geometry. Todo
        """
        self.clearMainDisplay()
        tk.Label(self.mainDisplay, text="Coming soon")\
                .place(relx=0.5, rely=0.5, anchor='center')

    def showLogWin(self):
        """
        Creates widgets associated with the Log window.
        This window is used to show main commands used in the past and
        system responses. Todo
        """
        self.clearMainDisplay()
        tk.Label(self.mainDisplay, text="Coming soon: log window")\
                .place(relx=0.5, rely=0.5, anchor='center')

    def getTemperature(self, tempLbl):
        """
        Starts temperature calculation function in the following steps:
        1. Select images to use for temperature calculations (suggest > 5)
        2. Select background image of just probe
        3. Sends filepaths to getTempFromImgList and returns temperature to
           display on label

        :param tempLbl: (tkinter.Label) lable to display data on

        :return: None
        """
        intialDir = r"./saved_Images"

        # fd brings up windows explorer to find image files
        fileNames = fd.askopenfilenames(initialdir=intialDir,
                                        title="Select MOT image files")

        bgImgPath = fd.askopenfilename(initialdir=intialDir,
                                       title="Select background image")

        # Perform calculations if user selected files
        if len(fileNames) > 2 and bgImgPath:
            # Get Temperature from selected images
            T = motTemperature.getTempFromImgList(fileNames, bgImgPath)

            # Convert T to string and reduce significant figures
            self.temperature = "{0:.4e}".format(T)

            # Update label temperature display
            tempLbl.configure(text=self.temperature)

        # Exit funciton if user decided to cancel temp calculations
        else:
            if len(fileNames) <= 2:
                print("Please select more than 2 images for more accurate "
                      "temperature calculation")
                print("Cancelling temperature calculation")

            return

    def getNumAtomsAbs(self, numAtomsLbl):
        """
        Starts temperature calculation function in the following steps:
        1. Select MOT image (suggest select one of the earlier images)
        2. Select background image of just probe
        3. Select dark bg image
        4. Sends filepaths to numAtomsAbs and returns #atoms calculated by
        absorption to display on label

        :param numAtomsLbl: (tkinter.Label) lable to display data on
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

        # Only perform calculations if user input paths
        if motImgPath and probeImgPath and bgImgPath:
            self.numAtomsAbs = numAtoms.numAtomsAbs([motImgPath],
                                                          [probeImgPath],
                                                          [bgImgPath])
            self.numAtomsAbs = "{0:.4e}".format(self.numAtomsAbs)

            numAtomsLbl.configure(text=self.numAtomsAbs)

        else:
            return
            
    def calibrateCameraShutter(self):
        """
        Calibrates shutter speed for cam1, and assigns the same shutter
        speed value to cam0
        """
        shutterSpeed = self.cam1.calibrateShutterSpeed()
        self.cam0.shutter_speed = shutterSpeed
        
        

    def snapImages(self, cam0Lbl, cam1Lbl):
        """
        Snaps new images to the designated labels
        cam0Lbl (tk.Label): Label to display image of cam0
        cam1Lbl (tk.Label): Label to display image of cam1
        """
        self.log.append("Snapped Images")
        threading.Thread(target=lambda: self.cam0.showImgOnLbl(cam0Lbl)).start()
        threading.Thread(target=lambda: self.cam1.showImgOnLbl(cam1Lbl)).start()

    def saveImage(self):
        """
        Saves the last images snapped by the cameras. Grab both images as
        np.arrays and stack them veritcally. Finally saving them by their
        timestamp in the output folder.
        """
        # Create timestamp and file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        savePath = f"{self.output}/{timestamp}.jpg"
        
        # Stack cam0 and cam1 images vertically
        combinedImg = np.concatenate((self.cam1.img, self.cam0.img), axis=0)
        
        # If not grayscale, image need to reorder image dimensions to match RGB
        if combinedImg.shape == (272*2, 608, 3):
            combinedImg = combinedImg[:, :, ::-1]
            
        # Save combined image
        cv2.imwrite(savePath, combinedImg)
        
        print(f"Image saved at {savePath}")

    def clearMainDisplay(self):
        """
        Clears all widgets on the display to make room on the new window.
        Called at the start of every new window call
        """
        for widget in self.mainDisplay.winfo_children():
            widget.destroy()

    def onWinClose(self):
        """
        Callback for when the gui is closed. Need to ensure the camera
        resources are released
        """
        print("Exiting Application")

        # Release camera resources
        if self.camOn:
            self.cam0.close()
            self.cam1.close()

        self.master.destroy()

def resizeImage(imgPath, h, w):
    """
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


if __name__ == "__main__":
    window = tk.Tk()
    window.title('PiCamera')
    
    gui = PiCameraGUI(window, debug=False, camOn=True)
    
    window.protocol("WM_DELETE_WINDOW", gui.onWinClose)
    gui.pack()
    
    window.mainloop()
