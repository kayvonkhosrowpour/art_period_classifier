import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import fft
from numpy.fft import fft2, fftshift, ifft, ifftshift, ifft2, ifftn
from skimage.feature import greycomatrix, greycoprops
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
from matplotlib import cm
from PIL import Image, ImageFilter, ImageOps
import scipy.stats as stats
import scipy.signal as signal

def auto_canny(image, sigma):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))

	# return the edged image
	return lower, upper


def average_edges(img, sigma):
    lower, upper = auto_canny(img, sigma)
    edges = cv2.Canny(img,lower,upper)
    x,y = img.shape
    count = 0
    for i in range(0,x):
        for j in range(0,y):
            if img[i,j] == 1:
                count = count + 1
    percentage = count / (x*y)

    return percentage

def x_y_gradient(img):
    x,y = img.shape
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    countx = 0
    county = 1
    for i in range(0,x):
        for j in range(0,y):
            if sobelx[i,j] == 1:
                countx = countx + 1
            if sobely[i,j] == 1:
                county = county + 1
    percentage = countx / (x*y)
    percentage2 = county / (x*y)
    return percentage, percentage2

def low_pass_filter(img, th1):
    width, height = img.shape
    fft = np.fft.fftshift(np.fft.fft2(img))
    h_Filter_Low_Pass = np.zeros(img.size, img.dtype).reshape(img.shape)
    for i in range(0, width):
        for j in range(0, height):
            if ((i - width/2)**2 + (j - height/2)**2) < th1**2:
                h_Filter_Low_Pass[i, j] = 1
    #g = fft * h_Filter_Low_Pass
    g = np.multiply(fft, h_Filter_Low_Pass)
    g_ifft = (np.fft.ifft2(np.fft.ifftshift(g)).real)
    sum = 0
    for i in range(0, width):
        for j in range(0, height):
            sum = sum + g_ifft[i,j]**2
    energy = np.sqrt(sum)
    median = np.median(g_ifft)
    deviation = np.std(g_ifft)
    return energy, median, deviation

def high_pass_filter(img, th1):
    width, height = img.shape
    fft = np.fft.fftshift(np.fft.fft2(img))
    h_Filter_Low_Pass = np.zeros(img.size, img.dtype).reshape(img.shape)
    for i in range(0, width):
        for j in range(0, height):
            if ((i - width/2)**2 + (j - height/2)**2) > th1**2:
                h_Filter_Low_Pass[i, j] = 1
    #g = fft * h_Filter_Low_Pass
    g = np.multiply(fft, h_Filter_Low_Pass)
    g_ifft = (np.fft.ifft2(np.fft.ifftshift(g)).real)
    sum = 0
    for i in range(0, width):
        for j in range(0, height):
            sum = sum + g_ifft[i,j]**2
    energy = np.sqrt(sum)
    median = np.median(g_ifft)
    deviation = np.std(g_ifft)
    return energy, median, deviation

def band_pass_filter(img, th1, th2):
        width, height = img.shape
        print(width,height)
        fft = (np.fft.fftshift(np.fft.fft2(img)))
        h_Filter_High_Pass = np.zeros(img.size, img.dtype).reshape(img.shape)
        for i in range(0, width):
            for j in range(0, height):
                if ((i - width/2)**2 + (j - height/2)**2) > th1**2:
                    h_Filter_High_Pass[i, j] = 1
        g = fft * h_Filter_High_Pass

        h_Filter_Low_Pass = np.zeros(img.size, img.dtype).reshape(img.shape)
        for i in range(0, width):
            for j in range(0, height):
                if ((i - width/2)**2 + (j - height/2)**2) < th2**2:
                    h_Filter_Low_Pass[i, j] = 1
        g2 = g * h_Filter_Low_Pass
        g_ifft = (np.fft.ifft2(np.fft.ifftshift(g2)).real)
        sum = 0
        for i in range(0, width):
            for j in range(0, height):
                sum = sum + g_ifft[i,j]**2
        energy = np.sqrt(sum)
        median = np.median(g_ifft)
        deviation = np.std(g_ifft)
        return energy, median, deviation


def stat_features(img):
    std_dev =  np.std(img)
    kurtosis = stats.kurtosis(img)
    skew = stats.skew(img)
    return std_dev, kurtosis, skew
