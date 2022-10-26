import numpy as np
import cv2
from imutils.video import VideoStream
import imutils
import time
import os

cap = cv2.VideoCapture("inputs/video_test_extract.mp4")
result = cv2.VideoWriter('result.mp4', -1, 20.0, (640, 480))

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
    detailenhanced = cv2.detailEnhance(sky, sigma_s=10, sigma_r=0.15)

    # Grayscale
    gray = cv2.cvtColor(detailenhanced, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Edge detection
    low_threshold = 50
    high_threshold = 150
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

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
result.release()
cv2.destroyAllWindows()
