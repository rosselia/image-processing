import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# warning
warning = cv2.imread("warning.png")
cv2.imshow('warning !', warning)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 1 - 2 MATLAB
# 3 - 4
# we used rotating formula to do this method. it works with parameters, so it can be work for any degree.
def naive_image_rotate(image, degree):
    rads = math.radians(degree)

    height_rot_img = round(abs(image.shape[0] * math.cos(rads))) + \
                     round(abs(image.shape[1] * math.sin(rads)))
    width_rot_img = round(abs(image.shape[1] * math.cos(rads))) + \
                    round(abs(image.shape[0] * math.sin(rads)))

    rot_img = np.uint8(np.zeros((height_rot_img, width_rot_img, image.shape[2])))
    cx, cy = (image.shape[1] // 2, image.shape[0] // 2)

    midx, midy = (width_rot_img // 2, height_rot_img // 2)

    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            x = (i - midx) * math.cos(rads) + (j - midy) * math.sin(rads)
            y = -(i - midx) * math.sin(rads) + (j - midy) * math.cos(rads)

            x = round(x) + cy
            y = round(y) + cx

            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                rot_img[i, j, :] = image[x, y, :]

    return rot_img


image = cv2.imread("city.jpg")
rotated_image = naive_image_rotate(image, 90)
cv2.imshow("original image", image)
cv2.imshow("image rotated 90 degrees clockwise", rotated_image)
rotated_image2 = naive_image_rotate(image, 270)
cv2.imshow("image rotated 90 degrees counterclockwise", rotated_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5 - resize
# first, we transformed our photo to gray scale because it must be a two-dimensional array. then we separated
# image to rows and columns, and we took each 4 pixels from image then calculated average of them. then we took the
# average to the brand-new array and made the new picture

image = cv2.imread('city.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = np.array(image)
col, row = img.shape
list_of_last = []
i = 0
while i <= (row // 2):
    r_iter = 0
    for a, val1 in enumerate(img):
        arr_split = img[r_iter:r_iter + 2, :]
        r_iter += 2
        i += 1

        if i > row:
            break
        n_iter = 0
        i2 = 0
        for j, val2 in enumerate(img):
            if i2 < (col // 2):
                nn = n_iter + 2
                arr_split_col = arr_split[:, n_iter:nn]
                list_of_last.append(((arr_split_col).sum()) // 4)
                i2 += 1
                n_iter += 2

                if i2 > col:
                    idx = 0
                    idx += 1
                    break

nmp = np.array(list_of_last)
nmp = nmp[nmp != 0]
nmp = nmp.reshape((row // 2), (col // 2))
cv2.imshow("not resized (original)", img)
img = nmp
img = img.astype(np.uint8)
cv2.imshow("resized by the half", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 6 - MATLAB
# 7
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


img = cv2.imread('city.jpg')
gammaImg = gammaCorrection(img, 2.2)

cv2.imshow('Original image', img)
cv2.imshow('Gamma corrected image', gammaImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8
img = plt.imread('city.jpg')
img_flat = img.flatten()
plt.hist(img_flat, bins=200, range=[0, 256])
plt.title("Histogram of the image")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
