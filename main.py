from motCamera import MOTCamera
from datetime import datetime
from PIL import ImageTk, Image

import tkinter as tk
import tkinter.font as tkFont
import numpy as np
import threading
import cv2

class PiCameraGUI(tk.Frame):
    def __init__(self, master, output=r"./saved_images", debug=False):
        self.master = master
        self.output = output
        self.debug = debug
        self.mWidth = 800  # Set main window width
        self.mHeight = 600  # Set main window height
        self.defaultFont = 'Courier'  # Default font style
        self.cam0 = MOTCamera(1, grayscale=True)  # Pi compute module rease cam0 port as numerical value 1
        self.cam1 = MOTCamera(0, grayscale=True)  # Pi compute module rease cam1 port as numerical value 0
        
        tk.Frame.__init__(self, master)
        
        # Create main panel and show
        self.mainDisplay = tk.Frame(master=self.master, height=self.mHeight, width=self.mWidth,
                                    highlightbackground="black", highlightthicknes=1)
        self.mainDisplay.pack()

        # Default starting window
        self.showCameraWin()

        # Add the navigation buttons at the bottom
        self.createNavigationBtn()

    def createNavigationBtn(self):
        '''
        Creates + displays the bottom navigation buttons and assigns 
        their respective window display functions
        '''
        btnHeight = 70
        btnWidth = 160

        btnFonts = tkFont.Font(family=self.defaultFont, size=15)

        navigationFrame = tk.Frame(master=self.master, height=btnHeight+2, width=self.mWidth,
                                   highlightbackground="black", highlightthicknes=1)

        # Create Buttons #
        alignmentBtn = tk.Button(navigationFrame, text='Alignment', font=btnFonts, command=self.showAlignmentWin)
        alignmentBtn.place(x=0, y=0, height=btnHeight, width=btnWidth)

        analysisBtn = tk.Button(navigationFrame, text='Analysis', font=btnFonts, command=self.showAnalysisWin)
        analysisBtn.place(x=btnWidth, y=0, height=btnHeight, width=btnWidth)

        cameraBtn = tk.Button(navigationFrame, text='Camera View', font=btnFonts, command=self.showCameraWin)
        cameraBtn.place(x=btnWidth*2, y=0, height=btnHeight, width=btnWidth)
        
        viewBtn = tk.Button(navigationFrame, text='3D View', font=btnFonts, command=self.show3DWin)
        viewBtn.place(x=btnWidth*3, y=0, height=btnHeight, width=btnWidth)
        
        logBtn = tk.Button(navigationFrame, text='Log', font=btnFonts, command=self.showLogWin)
        logBtn.place(x=btnWidth*4, y=0, height=btnHeight, width=btnWidth)

        navigationFrame.pack()

    def showAlignmentWin(self):
        '''
        Creates widgets associated with the alignment window. 
        This window is used to see how the MOT is aligned compared to the fiber
        '''
        self.clearMainDisplay()

        # Create right side information panel #
        coordinatesFrame = tk.Frame(self.mainDisplay, height=570, width=225,
                                    highlightbackground="black", highlightthicknes=1)
        coordinatesFrame.place(x=570, y=11)

        motLblFont = tkFont.Font(family=self.defaultFont, size=20)
        tk.Label(coordinatesFrame, text='MOT Position', font=motLblFont)\
            .place(relx=0.5, rely=0.05, anchor='center')

        # Get position and number of atoms data
        # Todo get real atom cloud position
        positions = np.array([-12, -5, 10])
        numAtoms = 101367

        # Display Data
        dataFont = tkFont.Font(family=self.defaultFont, size=20)
        relYStart = 0.18
        relDist = 0.225  # increments in rely between parameters
        tk.Label(coordinatesFrame, text=f'x(nm)\n{positions[0]}', font=dataFont)\
            .place(relx=0.5, rely=relYStart, anchor='center')
        tk.Label(coordinatesFrame, text=f'y(nm)\n{positions[1]}', font=dataFont)\
            .place(relx=0.5, rely=relYStart + relDist, anchor='center')
        tk.Label(coordinatesFrame, text=f'z(nm)\n{positions[2]}', font=dataFont)\
            .place(relx=0.5, rely=relYStart + relDist * 2, anchor='center')
        tk.Label(coordinatesFrame, text=f'#Atoms\n{numAtoms}', font=dataFont)\
            .place(relx=0.5, rely=relYStart + relDist * 3, anchor='center')

        # Draw MOT on grid #
        gHeight = 570  # Grid height
        gWidth = 550  # Grid width
        alignmentGrid = tk.Canvas(self.mainDisplay, bg="gray60", height=gHeight, width=gWidth)
        alignmentGrid.place(x=15, y=10)

        # Draw axis lines
        alignmentGrid.create_line(0, gHeight/2, gWidth, gHeight/2)
        alignmentGrid.create_line(gWidth/2, 0, gWidth/2, gHeight)

        # Scale pixel location to real location
        zoom = 15  # Todo dynamic scalable dimensions
        positions = positions*zoom
        # Draw MOT
        motRadius = 15  # Todo dynamic radius?
        alignmentGrid.create_oval(gWidth/2+positions[0]-motRadius,
                                  gHeight/2-positions[1]-motRadius,
                                  gWidth/2+positions[0]+motRadius,
                                  gHeight/2-positions[1]+motRadius,
                                  fill='slate gray')

    def showAnalysisWin(self):
        '''
        Creates widgets associated with the Analysis window. 
        This window is used to calculate temperature and #Atoms
        '''
        self.clearMainDisplay()

        # Set fonts to be used in labels
        lblFont = tkFont.Font(family=self.defaultFont, size=25)
        dataFont = tkFont.Font(family=self.defaultFont, size=30)

        # Todo get real temperature and atom count
        temp = 3.2
        numAtoms = 101367  # Todo Should probably be a field instead

        # Display temperature data
        tk.Label(self.mainDisplay, text=f'Temperature (mK)', font=lblFont)\
            .place(x=self.mWidth*3/10, y=self.mHeight/2-30, anchor='center')
        tk.Label(self.mainDisplay, text=f'{temp}', font=dataFont)\
            .place(x=self.mWidth*3/10, y=self.mHeight/2+30, anchor='center')

        # Display atom count data
        tk.Label(self.mainDisplay, text=f'#Atoms', font=lblFont)\
            .place(x=self.mWidth*7/10, y=self.mHeight/2-30, anchor='center')
        tk.Label(self.mainDisplay, text=f'{numAtoms}', font=dataFont)\
            .place(x=self.mWidth*7/10, y=self.mHeight/2+30, anchor='center')

    def showCameraWin(self):
        '''
        Creates widgets associated with the Camera window. 
        This window is used to snap and save images from the camera and
        create the video view popout window
        ''' 
        self.clearMainDisplay()
        
        # Main camera Displays #
        camDispHeight = 272;
        camDispWidth = 608;

        cam0Lbl = tk.Label(self.mainDisplay, height=camDispHeight, width=camDispWidth, bd=1, relief='solid')
        cam0Lbl.place(x=40, y=25)
        threading.Thread(target=lambda : self.cam0.showImgOnLbl(cam0Lbl)).start() # Snap an image upon opening window

        cam1Lbl = tk.Label(self.mainDisplay, height=camDispHeight, width=camDispWidth, bd=1, relief='solid')
        cam1Lbl.place(x=40, y=300)
        threading.Thread(target=lambda : self.cam1.showImgOnLbl(cam1Lbl)).start() # Snap an image upon opening window

        ## Camera Labels ##
        btnRelx = 0.91
        distY = 60
        btnH = 60;
        btnW = 65;

        camFont = tkFont.Font(family=self.defaultFont, size=13)
        tk.Label(self.mainDisplay, text='cam0', font=camFont, bg='gray83')\
            .place(x=41, y=26)
        tk.Label(self.mainDisplay, text='cam1', font=camFont, bg='gray83')\
            .place(x=41, y=301)

        ## Video Button for cam 0 ##
        vidImgPath = r'./assets/vid_0.png'
        vidImg = resizeImage(vidImgPath, btnH, btnW)
        vidBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE,
                           command=lambda: threading.Thread(target=self.cam0.showVid).start() )
        vidBtn.image = vidImg
        vidBtn.configure(image=vidImg)
        vidBtn.place(relx=btnRelx, y=self.mHeight/2-3*distY, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Show Vid 0').place(relx=btnRelx, y=105, anchor='center')

        ## Snap Image Button ##
        snapImgPath = r'./assets/snap.png'
        snapImg = resizeImage(snapImgPath, btnH, btnW)
        snapBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE, 
                            command=lambda : self.snapImages(cam0Lbl, cam1Lbl))
        snapBtn.image = snapImg
        snapBtn.configure(image=snapImg)
        snapBtn.place(relx=btnRelx, y=self.mHeight/2-distY, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Snap Pictures').place(relx=btnRelx, y=259, anchor='center')
            
        ## Save Button ##
        saveImgPath = r'./assets/save.png'
        saveImg = resizeImage(saveImgPath, btnH, btnW)
        saveBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE,
                            command=self.saveImage)
        saveBtn.image = saveImg
        saveBtn.configure(image=saveImg)
        saveBtn.place(relx=btnRelx, y=self.mHeight/2+distY, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Save Images').place(relx=btnRelx, y=413, anchor='center')
            
            
        ## Video Button for cam1 ##
        vidImgPath = r'./assets/vid_1.png'
        vidImg = resizeImage(vidImgPath, btnH, btnW)
        vidBtn = tk.Button(self.mainDisplay, relief=tk.GROOVE,
                           command=lambda: threading.Thread(target=self.cam1.showVid).start() )
        vidBtn.image = vidImg
        vidBtn.configure(image=vidImg)
        vidBtn.place(relx=btnRelx, y=self.mHeight/2+3*distY, anchor='center')
        # ~ tk.Label(self.mainDisplay, text='Show Vid 1').place(relx=btnRelx, y=567, anchor='center')
        
    def show3DWin(self):
        '''
        Creates widgets associated with the 3D window. 
        This window is used to show a 3D view of the MOT based on two-view
        geometry. Todo
        ''' 
        self.clearMainDisplay()
        tk.Label(self.mainDisplay, text="Coming soon: matplotlib 3D view of structure")\
                .place(relx=0.5, rely=0.5, anchor='center')
                
    def showLogWin(self):
        '''
        Creates widgets associated with the Log window. 
        This window is used to show main commands used in the past and
        system responses. Todo
        ''' 
        self.clearMainDisplay()
        tk.Label(self.mainDisplay, text="Coming soon: log window")\
                .place(relx=0.5, rely=0.5, anchor='center')


    def snapImages(self, cam0Lbl, cam1Lbl):
        '''
        Snaps new images to the designated labels
        cam0Lbl (tk.Label): Label to display image of cam0
        cam1Lbl (tk.Label): Label to display image of cam1
        '''
        
        threading.Thread(target=lambda : self.cam0.showImgOnLbl(cam0Lbl)).start()  
        threading.Thread(target=lambda : self.cam1.showImgOnLbl(cam1Lbl)).start()  

    def saveImage(self):
        '''
        Saves the last images snapped by the cameras. Grab both images as
        np.arrays and stack them veritcally. Finally saving them by their
        timestamp in the output folder.
        '''
        # Create timestamp and file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        savePath = f"{self.output}/{timestamp}.jpg"
        
        # Stack cam0 and cam1 images vertically
        combinedImg = np.concatenate((self.cam1.img, self.cam0.img), axis=0)
        
        # Save combined image
        cv2.imwrite(savePath, combinedImg)
        
        print(f"Image saved at {savePath}")

    def clearMainDisplay(self):
        '''
        Clears all widgets on the display to make room on the new window. 
        Called at the start of every new window
        '''
        for widget in self.mainDisplay.winfo_children():
            widget.destroy()
            
    def onWinClose(self):
        '''
        Callback for when the gui is closed, need to ensure the camera 
        resources are released
        '''
        print("Exiting Application")
        self.cam0.close()
        self.cam1.close()

        self.master.destroy()

def resizeImage(imgPath, h, w):
    '''
    Maps image to the dimensions of h and w and then returns it as an 
    ImageTk.PhotoImage object to use in tkinter widgets
    
    Inputs
    imgPath (str) = file path to image
    h (int) = desired image height in pixels
    w (int) = desired image width in pixels
    
    Output
    img (ImageTk.PhotoImage) = image to use in tkinter widgets
    '''
    img = Image.open(imgPath)
    img = img.resize((w, h), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    return img

if __name__ == "__main__":
    window = tk.Tk()
    window.title('PiCamera')
    
    gui = PiCameraGUI(window, debug=False)
    
    window.protocol("WM_DELETE_WINDOW", gui.onWinClose)
    gui.pack()
    
    window.mainloop()
