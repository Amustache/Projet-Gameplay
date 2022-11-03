import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture("../inputs/video_test_extract.mp4")
ret, frame_0 = cap.read()
if not ret:
    exit(-1)


def extract_enhance_minimap(frame):
    minimap = frame[750:, 1500:]
    h, w, _ = minimap.shape

    mask = cv2.imread('../inputs/mask.png', 0)
    mask = cv2.resize(mask, (w, h))

    minimap_masked = cv2.bitwise_and(minimap, minimap, mask=mask)

    enhanced = cv2.detailEnhance(minimap_masked, sigma_s=25, sigma_r=0.15)

    return enhanced


while True:
    ret, frame_1 = cap.read()
    if not ret:
        exit(-1)

    minimap_0 = extract_enhance_minimap(frame_0)
    minimap_1 = extract_enhance_minimap(frame_1)

    cv2.imshow('Minimap 0', minimap_0)
    cv2.imshow('Minimap 1', minimap_1)

    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(minimap_0, None)
    kp2, des2 = sift.detectAndCompute(minimap_1, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, _ = minimap_0.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        minimap_1 = cv2.polylines(minimap_1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(minimap_0, kp1, minimap_1, kp2, good, None, **draw_params)
    cv2.imshow('Comparison', img3)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    frame_0 = frame_1

cap.release()
cv2.destroyAllWindows()
