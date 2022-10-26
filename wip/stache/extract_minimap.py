import numpy as np
import cv2
from imutils.video import VideoStream
import imutils
import time
import os

cap = cv2.VideoCapture("inputs/video_test_extract.mp4")
# result = cv2.VideoWriter('result.mp4', -1, 20.0, (640, 480))

SR = cv2.dnn_superres.DnnSuperResImpl_create()


def init_super(model, base_path='models'):
    # Define model path
    model_path = os.path.join(base_path, model + ".pb")

    # Extract model name from model path
    model_name = model.split('_')[0].lower()

    # Extract model scale from model path
    model_scale = int(model.split("_")[1][1])

    # Read the desired model
    SR.readModel(model_path)

    SR.setModel(model_name, model_scale)


def super_res(image):
    # Upscale the image
    final_img = SR.upsample(image)

    return final_img


init_super("ESPCN_x4")
# init_super("EDSR_x4")


low_threshold = 50
high_threshold = 150
sigma_s = 25
sigma_r = 0.15


# Parameters
def on_change_low(value):
    global low_threshold
    low_threshold = value


def on_change_high(value):
    global high_threshold
    high_threshold = value


def on_change_sigma_s(value):
    global sigma_s
    sigma_s = value


def on_change_sigma_r(value):
    global sigma_r
    sigma_r = value


windowName = 'lines_finder'
cv2.namedWindow(windowName)
cv2.createTrackbar('low_threshold', windowName, 0, 500, on_change_low)
cv2.createTrackbar('high_threshold', windowName, 0, 500, on_change_high)
# cv2.createTrackbar('sigma_s', windowName, 0, 50, on_change_sigma_s)
# cv2.createTrackbar('sigma_r', windowName, 0, 1, on_change_sigma_r)

while True:
    # First, we get the pic
    ret, frame = cap.read()
    if not ret:
        print("Error, could not read file")
        exit(-1)

    sky = frame[750:, 1500:]
    h, w, _ = sky.shape

    # Creating the mask, and keeping ground truth
    mask = cv2.imread('inputs/mask.png', 0)
    mask = cv2.resize(mask, (w, h))
    res = cv2.bitwise_and(sky, sky, mask=mask)

    # Enhance resolution
    # dst = super_res(sky)
    detailenhanced = cv2.detailEnhance(sky, sigma_s=sigma_s, sigma_r=sigma_r)

    # Grayscale
    gray = cv2.cvtColor(detailenhanced, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Edge detection
    # low_threshold = 50
    # high_threshold = 150
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Resize
    result = cv2.resize(edges, (w, h))

    # Mask
    result = cv2.bitwise_and(result, result, mask=mask)

    # Show and compare results
    cv2.imshow('Original', sky)
    cv2.imshow('More details', detailenhanced)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Blured', blur_gray)
    cv2.imshow('Edges', edges)
    cv2.imshow('Result', result)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(sky) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(sky, 0.8, line_image, 1, 0)

    cv2.imshow(windowName, lines_edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
result.release()
cv2.destroyAllWindows()
