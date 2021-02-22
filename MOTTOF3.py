#############################################################################
#############################################################################
# Magneto Optical Trap Time of Flight Analyzer
#
# Updated: Feb. 13, 2017
# Coded by Turner Silverthorne
# Property of the Bajcsy NPQO Group
# Institute for Quantum Computing
#
#############################################################################
#############################################################################

# DEBUGGING FLAGS
importflag = True
centerfindflag = True
gaussianfitflag = True
visualflag = False
Gauss2Dflag = True

fitaxis = 0 	# 0: x-axis, 1: y-axis

cropx = 470     # center x position of the crop square 
cropy = 400     # center y position of the crop square
cropsize = 350  # Half of crop square dimension

rcropx = 460    # Crop for background intensity comparison
rcropy = 480
rcropsize = 300  # Half of rcrop square dimension

# IMPORT STATEMENTS
import numpy
from PIL import Image, ImageFilter
import glob
from pylab import imshow, show, gray
from matplotlib import pyplot
from matplotlib import mlab
from scipy.optimize import curve_fit
from numpy import linspace

# FUNCTION DECLARATIONS

# crops images
# def crop(image,rmin,rmax,cmin, cmax): # r for row, c for column
#	return image[rmin:rmax,cmin:cmax]

# to be used in offsetting background from MOT image
# you can tune them if you want, but I usually leave them at 0
delx = 0
dely = 0

# Takes in MOT image and background image, and outputs the sigma value
# of the log of intensity (along 1-Dimensional cross section)
def image_to_sigma(imgback, imgfore):
	# Crop the images and take their difference
	ratio = numpy.sum(imgback[rcropy-rcropsize:rcropy+rcropsize,rcropx-rcropsize:rcropx+rcropsize])/numpy.sum(imgfore[rcropy-rcropsize+dely:rcropy+rcropsize+dely,rcropx-rcropsize+delx:rcropx+rcropsize+delx])
	imdif = numpy.array(imgback[cropy-cropsize:cropy+cropsize,cropx-cropsize:cropx+cropsize] - ratio*imgfore[cropy-cropsize+dely:cropy+cropsize+dely,cropx-cropsize+delx:cropx+cropsize+delx],'float')
	(X,Y) = numpy.shape(imdif)

	# find center of brightness
	if centerfindflag == True:
		# loop to find center of image
		thresh =25 
		m = numpy.zeros((X,Y))
	
		for x in range(X):
			for y in range(Y):
        			m[x, y] = imdif[x, y] >= thresh
		m = m / numpy.sum(numpy.sum(m))
		
		# marginal distributions
		dx = numpy.sum(m, 1)
		dy = numpy.sum(m, 0)
	
		# expected values
		cx = numpy.sum(dx * numpy.arange(X))
		cy = numpy.sum(dy * numpy.arange(Y))
		
		# optional visualization
		if visualflag == True:
			pyplot.figure(1,figsize=(4,8))
			pyplot.subplot(211)
			pyplot.imshow(imdif) 
			if fitaxis == 0:
			    pyplot.plot([0,2*cropsize],[cx,cx],'b-',lw=2)
			elif fitaxis == 1:
			    pyplot.plot([cy,cy],[0,2*cropsize],'b-',lw=2)

			pyplot.plot(cy,cx,'bo')
			gray()
			pyplot.gcf()
	
		
	# take horizontal and vertical cuts of I_bck - I_mot
	# and also I_bck
	imref_full = numpy.array(imgback[cropy-cropsize:cropy+cropsize,cropx-cropsize:cropx+cropsize],'float')
	if fitaxis == 0:
	    imref_cut = numpy.array(imref_full[int(cx),:], 'float')
	    samp = numpy.array(imdif[int(cx),:],'float')
	elif fitaxis == 1:
	    imref_cut = numpy.array(imref_full[:,int(cy)], 'float')
	    samp = numpy.array(imdif[:,int(cy),],'float')
	
	x = numpy.arange(len(samp))
	z = numpy.zeros(len(samp))
		
	# take log only at pixel values where log is well defined
	for i in range(len(samp)):
		if samp[i]/imref_cut[i]< 1 and imref_cut[i] > 0 and samp[i] >0:
			z[i] = -numpy.log(1.0000-samp[i]/imref_cut[i])

	if Gauss2Dflag == True:
		# xx = numpy.mgrid[0:X+0.1:1, 0:Y+0.1:1].reshape(2,-1).T
		zz = numpy.zeros([X,Y])
		for i in range(X):
			for j in range(Y):
				if imdif[i,j]/imref_full[i,j] and imdif[i,j] > 0:
					zz[i,j] = -numpy.log(1.0000-imdif[i,j]/imref_full[i,j])
		zz1d = zz.ravel()
                 
					
	# find average and sigma
	mean = sum(x * z) / sum(z)
	sigma = numpy.sqrt(sum(z * (x - mean)**2) / sum(z))
	ambient = 0.01
	
	if Gauss2Dflag == False:
		# define curve to be fitted
		def Gauss(x, a, x0, sigma, b):
			return a * numpy.exp(-(x - x0)**2 / (2 * sigma**2)) + b

		# fit curve
		popt,pcov = curve_fit(Gauss, x, z, p0=[max(z), mean, sigma, ambient])
	
	# define curve to be fitted
	else:
		def Gauss2D(XX, a, x0, y0, sigma, b):
			val = a * numpy.exp(-((XX[0]-x0)**2 + (XX[1]-y0)**2) / (2*sigma**2)) + b
			return val.ravel()

                   
		# create x and y indices
		xx = numpy.linspace(0, X-1, X)
		yy = numpy.linspace(0, Y-1, Y)
		xx, yy = numpy.meshgrid(xx, yy)

		popt,pcov = curve_fit(Gauss2D, (xx,yy), zz1d, p0=[numpy.amax(zz), cx, cy, sigma, ambient])
		
	# optional visualization
	if visualflag == True:
		pyplot.figure(1)
		pyplot.subplot(212)
		if Gauss2Dflag == False:
			pyplot.plot(x, z, 'b+:', label='data')
			pyplot.plot(x, Gauss(x, *popt), 'r-', label='fit')
			pyplot.legend()
		pyplot.title('Gaussian Fit Trial')
		pyplot.xlabel('Pixel Number')
		pyplot.ylabel('Intensity 0 - 255')
		pyplot.gcf
		pyplot.show()
		
	if Gauss2Dflag == False:
		print('sigma = {0:.6f},\t popt[2] = {1:.6f}'.format(sigma, popt[2]))
		return popt[2]
	else:
		print('sigma = {0:.6f},\t popt[3] = {1:.6f}'.format(sigma, popt[3]))
		return popt[3]

# IMAGE IMPORT STAGE
if importflag == True:
	# look in directory, import all jpegs as one array
	# converts them to greyscale 0-255
	#filelist = glob.glob('/Users/Admin/Research/2017/MOTWORK/RawData/*.jpg')
	
	#I am using manual importing of files because they will otherwise be imported in a 
	#non-chronological order 

	filelist = []
	filelist.append('04ms.png')
	filelist.append('06ms.png')
	filelist.append('08ms.png')
	filelist.append('10ms.png')
	filelist.append('12ms.png')
	filelist.append('14ms.png')
	filelist.append('16ms.png')
	filelist.append('18ms.png')
	filelist.append('20ms.png')
	filelist.append('22ms.png')
	# filelist.append('24ms.png')
	# filelist.append('26ms.png')
	# filelist.append('28ms.png')
	# filelist.append('bg.jpg')
	# filelist.append('500us.jpg')
	# filelist.append('01000us.jpg')
	# filelist.append('01500us.jpg')
	#filelist.append('02000us.jpg')
	# # filelist.append('02500us.jpg')
	# filelist.append('03000us.jpg')
	# # filelist.append('03500us.jpg')
	# filelist.append('05000us.jpg')
	# # filelist.append('04500us.jpg')
	# filelist.append('07000us.jpg')
	# # filelist.append('05500us.jpg')
	# filelist.append('09000us.jpg')
	# # filelist.append('06500us.jpg')
	# # filelist.append('11000us.jpg')
	# # filelist.append('07500us.jpg')
	# filelist.append('13000us.jpg')
	# # filelist.append('08500us.jpg')
	# filelist.append('15000us.jpg')
	# # filelist.append('09500us.jpg')
	# filelist.append('17000us.jpg')
	# # filelist.append('10500us.jpg')
	# filelist.append('19000us.jpg')
	# # filelist.append('11500us.jpg')
	# filelist.append('23000us.jpg')
	# # filelist.append('12500us.jpg')
	# filelist.append('25000us.jpg')
	# filelist.append('13500us.jpg')
	# filelist.append('27000us.jpg')
	# filelist.append('17000us.jpg')
	# filelist.append('18000us.jpg')
	# filelist.append('19000us.jpg')
	# filelist.append('20000us.jpg')
	#filelist.append('19000us.jpg')
	#filelist.append('20000us.jpg')
	#filelist.append('21000us.jpg')
	#filelist.append('22000us.jpg')
	#filelist.append('23000us.jpg')
	#filelist.append('24000us.jpg')
	#filelist.append('25000us.jpg')
	# imgarray = numpy.array([numpy.array(Image.open(fname).convert('L')) for fname in filelist])


# SIGMA VALUE FINDING STAGE
sig_ar = numpy.zeros(len(filelist)) # Array to store the sigma values
t_ar = numpy.zeros(len(filelist)) # time array
group = ".//legacy//Absorption images example//"
# group = ".//MOT2//"
bgarray = numpy.array(Image.open(group+'background.png').convert('L'))
i = 0
for filename in filelist:
	imgarray = numpy.array(Image.open(group+filename).convert('L'))
	sigma = image_to_sigma(bgarray, imgarray)
	sig_ar[i] = sigma
	t_ar[i] = float(filename[0:2])*10**-3
	i += 1
	del imgarray
	#pyplot.plot(3500 + 500*i,sigma,'ko')
# print('SIGMA VALS:')
print(sig_ar)

# LINEAR REGRESSION/TEMPERATURE OUTPUT STAGE
# sig_ar *= 2.5*0.0053*10**-3 # convert px number to meter
sig_ar *= 13.7*10**-6 # convert px number to meter

def fit_func(x, s, k): # function to fit to sigma array
	return numpy.sqrt(s**2 + k*x**2)

popt, pcov = curve_fit(fit_func, t_ar, sig_ar) # curve fitting
print(popt[0])

# Calculate R^2 (coefficient of determination)
residuals = sig_ar- fit_func(t_ar, popt[0], popt[1])
ss_res = numpy.sum(residuals**2)
ss_tot = numpy.sum((sig_ar-numpy.mean(sig_ar))**2)
r_squ = 1- (ss_res/ss_tot)

# Plot fit results
v_ar = numpy.linspace(min(t_ar),max(t_ar),1000)
yar = fit_func(v_ar,popt[0],popt[1])
pyplot.plot(1000*t_ar,sig_ar,'ko',ms=5)
pyplot.plot(1000*v_ar,yar,'b-',lw=3)
pyplot.xlabel('time [ms]', fontsize=18)
pyplot.ylabel('$\sigma$', fontsize=18)
pyplot.xticks(numpy.arange(0,26,2),fontsize=18)
pyplot.yticks([])
pyplot.tight_layout()
pyplot.show()

print('Coeff of corr: {0}'.format(r_squ))
print('Temperature: {0}'.format((popt[1]*2.206948425*10**-25)/(1.38*10**-23)))
