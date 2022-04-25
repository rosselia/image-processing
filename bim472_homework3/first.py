import numpy as np
import cv2

# reading the image
img = cv2.imread('redbird.jpg')

# convert image BGR to GRAYSCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# make GRAY image multichannel
gray = cv2.merge([gray, gray, gray])

# convert image BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# describe upper and lower boundaries for RED color
lower_bound1 = np.array([0, 195, 50])
upper_bound1 = np.array([10, 255, 255])

lower_bound2 = np.array([169, 195, 50])
upper_bound2 = np.array([180, 255, 255])

# make 2 masks
mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)

# add 2 mask together
mask = mask2+mask1

# apply mask to colored image
colored_output = cv2.bitwise_and(img, img, mask=mask)

# apply negative mask to GRAY image
gray_output = cv2.bitwise_and(gray, gray, mask=255 - mask)

# adding GRAY and colored image together
final = cv2.add(colored_output, gray_output)

cv2.imshow('final image', final)
cv2.imshow('original image', img)
cv2.waitKey()
