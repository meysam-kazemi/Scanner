# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

# =============================================================================
# Edge Detection
# =============================================================================
# load the image and compute the ratio of the old height to the new height, and resize it.
img = cv2.imread("image.jpg")
ratio = img.shape[0] / 500.0 # 500 : new height
original = img.copy()
img = imutils.resize(img,height=500) # resize the image

# Convert to gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur image
blur = cv2.GaussianBlur(gray,(5,5),0)
# Detecting edges
edge = cv2.Canny(gray,75,200)  
