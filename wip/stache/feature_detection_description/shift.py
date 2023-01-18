import cv2 as cv
import numpy as np


img = cv.imread("inputs/extracted_enhanced_minimap.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite("outputs/sift_keypoints.jpg", img)
