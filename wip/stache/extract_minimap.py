import numpy as np
import cv2

cap = cv2.VideoCapture("test_video_sample.mp4")
result = cv2.VideoWriter('result.mp4', -1, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (1920, 1080))
    sky = frame[750:, 1500:]
    h, w, _ = sky.shape
    sky = cv2.resize(sky, (w, h))

    mask = cv2.imread('mask.png', 0)
    mask = cv2.resize(mask, (w, h))

    res = cv2.bitwise_and(sky, sky, mask=mask)
    cv2.imshow('Video', res)
    result.write(sky)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
result.release()
cv2.destroyAllWindows()
