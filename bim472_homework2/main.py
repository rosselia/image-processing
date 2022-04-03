import cv2
import numpy as np

img = cv2.imread('city.jpg', 0)
# Obtain number of rows and columns of the image
m, n = img.shape

# Develop Averaging filter(3, 3) mask
mask = np.ones([3, 3], dtype=int)
mask = mask / 9

# Convolve the 3X3 mask over the image
img_new = np.zeros([m, n])

for i in range(1, m - 1):
    for j in range(1, n - 1):
        temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] + img[
            i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[i + 1, j - 1] * mask[
                   2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

        img_new[i, j] = temp

img_new = img_new.astype(np.uint8)

# Display
cv2.imshow('averaging filter - image made with our own algorithm', img_new)

# Where we apply function
cv2.imshow('averaging filter - image made with function', cv2.blur(img, (3, 3)))

# Where we apply function
res = cv2.equalizeHist(img)
cv2.imshow("histogram equalization - made with function", res)

# Code of Histogram Equalization Formula
arr = img
for i, val1 in enumerate(arr):
    for j, val2 in enumerate(val1):
        arr[i, j] = ((arr[i, j] ** 2) / 255)
final = arr

# Display
cv2.imshow("histogram equalization - made with our own algorithm", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
