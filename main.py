import tkinter as tk
import tkinter.font as tkFont
import numpy as np
from datetime import datetime
from PIL import ImageTk, Image



def resizeImage(imgPath, h, w):
    img = Image.open(imgPath)
    img = img.resize((w, h), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    return img


class PiCameraGUI(tk.Frame):
    def __init__(self, master, saveImgDir=r".\saved_images", debug=False):
        self.master = master
        self.debug = debug
        self.width = 800  # Set main window width
        self.mHeight = 600  # Set main window height
        self.defaultFont = 'Courier'  # Default font style, but size changes
        self.saveImageDir = saveImgDir
        tk.Frame.__init__(self, master)

        self.mainDisplay = tk.Frame(master=self.master, height=self.mHeight, width=self.width,
                                    highlightbackground="black", highlightthicknes=1)
        self.mainDisplay.pack()

        self.showCameraWin()

        self.createNavigationBtn()

    def createNavigationBtn(self):
        btnHeight = 80
        btnWidth = 150

        btnFonts = tkFont.Font(family=self.defaultFont, size=15)

        navigationFrame = tk.Frame(master=self.master, height=btnHeight+2, width=self.width,
                                   highlightbackground="black", highlightthicknes=1)

        alignmentBtn = tk.Button(navigationFrame, text='Alignment', font=btnFonts,
                                 command=self.showAlignmentWin)
        alignmentBtn.place(x=1, y=1, height=btnHeight, width=btnWidth)

        analysisBtn = tk.Button(navigationFrame, text='Analysis', font=btnFonts,
                                command=self.showAnalysisWin)
        analysisBtn.place(x=btnWidth+1, y=1, height=btnHeight, width=btnWidth)

        cameraBtn = tk.Button(navigationFrame, text='Camera View', font=btnFonts,
                              command=self.showCameraWin)
        cameraBtn.place(x=btnWidth*2+1, y=1, height=btnHeight, width=btnWidth)

        navigationFrame.pack()

    def showAlignmentWin(self):
        self.clearMainDisplay()

        # Create right side information panel #
        coordinatesFrame = tk.Frame(self.mainDisplay, height=570, width=225,
                                    highlightbackground="black", highlightthicknes=1)
        coordinatesFrame.place(x=570, y=12)

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
        self.clearMainDisplay()

        lblFont = tkFont.Font(family=self.defaultFont, size=25)
        dataFont = tkFont.Font(family=self.defaultFont, size=30)

        # Todo get real temperature and atom count
        temp = 3.2
        numAtoms = 101367  # Todo Should probably be a field instead

        # Display temperature data
        tk.Label(self.mainDisplay, text=f'Temperature (mK)', font=lblFont)\
            .place(x=self.width*3/10, y=self.mHeight/2-30, anchor='center')
        tk.Label(self.mainDisplay, text=f'{temp}', font=dataFont)\
            .place(x=self.width*3/10, y=self.mHeight/2+30, anchor='center')

        # Display atom count data
        tk.Label(self.mainDisplay, text=f'#Atoms', font=lblFont)\
            .place(x=self.width*7/10, y=self.mHeight/2-30, anchor='center')
        tk.Label(self.mainDisplay, text=f'{numAtoms}', font=dataFont)\
            .place(x=self.width*7/10, y=self.mHeight/2+30, anchor='center')

    def showCameraWin(self):
        self.clearMainDisplay()

        camDispHeight = 260;
        camDispWidth = 600;

        img1Path = r'./assets/img1.jpg'
        img2Path = r'./assets/img2.jpg'

        img1 = resizeImage(img1Path, camDispHeight, camDispWidth)
        img2 = resizeImage(img2Path, camDispHeight, camDispWidth)

        cam1 = tk.Label(self.mainDisplay, height=camDispHeight, width=camDispWidth, bd=1, relief='solid')
        cam1.image = img1
        cam1.configure(image=img1)
        cam1.place(x=40, y=25)

        cam2 = tk.Label(self.mainDisplay, height=camDispHeight, width=camDispWidth, bd=1, relief='solid')
        cam2.image = img2
        cam2.configure(image=img2)
        cam2.place(x=40, y=300)

        camFont = tkFont.Font(family=self.defaultFont, size=13)
        tk.Label(self.mainDisplay, text='cam0', font=camFont, bg='gray83')\
            .place(x=41, y=31)
        tk.Label(self.mainDisplay, text='cam1', font=camFont, bg='gray83')\
            .place(x=41, y=301)


        cameraImgPath = r'./assets/Capture.jpg'
        camImg = ImageTk.PhotoImage(Image.open(cameraImgPath))

        camBtn = tk.Button(self.mainDisplay, command=self.saveImage, relief=tk.GROOVE)
        camBtn.image = camImg
        camBtn.configure(image=camImg)
        camBtn.place(relx=0.9, y=290, anchor='center')
        tk.Label(self.mainDisplay, text='Save Image')\
            .place(relx=0.9, y=335, anchor='center')

    def saveImage(self):
        # Todo save image to some location
        identifier = datetime.now().strftime("%Y%m%d_%H%M%S")  # save images as title YearMonthDay_HourMinutesSecond
        print(identifier)

    def clearMainDisplay(self):
        for widget in self.mainDisplay.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    window.title('PiCamera')
    PiCameraGUI(window, debug=False).pack()
    window.mainloop()
