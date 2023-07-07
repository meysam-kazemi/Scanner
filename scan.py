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
img = cv2.imread("img.jpeg")
ratio = img.shape[0] / 500.0 # 500 : new height
original = img.copy()
img = imutils.resize(img,height=500) # resize the image

# Convert to gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur image
blur = cv2.GaussianBlur(gray,(5,5),0)
# Detecting edges
edged = cv2.Canny(gray,75,200)  

# # Show image and its edges
# cv2.imshow("Original image",img)
# cv2.imshow("Edges",edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# =============================================================================
# Finding Contours
# =============================================================================
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# # show the contour (outline) of the piece of paper
# print("STEP 2: Find contours of paper")
# cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# =============================================================================
# Apply a Perspective Transform & Threshold
# =============================================================================
# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)


