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

