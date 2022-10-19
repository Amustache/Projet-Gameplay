import cv2
import numpy as np
from matplotlib import pyplot as plt


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# reading image
img = cv2.imread('test_image2.png')
cv2.imshow('Result', img)
cv2.waitKey(0)

img = increase_brightness(img, value=100)
cv2.imshow('Result', img)
cv2.waitKey(0)

# Augment contrast
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(8, 8))
cl = clahe.apply(l_channel)
limg = cv2.merge((cl, a, b))
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# result = np.hstack((img, enhanced_img))
cv2.imshow('Result', enhanced_img)
cv2.waitKey(0)

# converting image into grayscale image
gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

# list for storing names of shapes
for i, contour in enumerate(contours):
    print("*" * 13)
    print(f"{i}/{len(contours)}")
    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        # putting shape name at center of each shape
        if len(approx) > 6:
            cv2.putText(img, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# displaying the image after drawing contours
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
