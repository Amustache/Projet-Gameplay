import csv
import itertools
import json

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def extract_enhance_minimap(frame, x_0=None, x_1=None, y_0=None, y_1=None):
    # frame = frame[750:, 1500:]
    frame = frame
    h, w, _ = frame.shape

    mask = cv2.imread('../inputs/mask_strict.png', 0)
    mask = cv2.resize(mask, (w, h))

    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    enhanced = cv2.detailEnhance(frame_masked, sigma_s=25, sigma_r=0.15)

    return enhanced


def sift_keypoints_and_descriptors(frame1, frame2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    return kp1, des1, kp2, des2


def sift_flann_matches(descriptors0, descriptors1):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors0, descriptors1, k=2)

    return matches


def sift_find_good_matches(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good


def sift_find_matching_keypoints(kp1, kp2, good):
    MIN_MATCH_COUNT = 5

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        src_pts, dst_pts, matchesMask = None, None, None

    return src_pts, dst_pts, matchesMask


def sift_comparison(frame1, kp1, frame2, kp2, good, matchesMask):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    comparison = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)

    return comparison


def main():
    cap = cv2.VideoCapture("../inputs/rotate.mp4")
    # cap = cv2.VideoCapture("../inputs/video_test_extract_2.mp4")
    ret, frame_0 = cap.read()
    if not ret:
        exit(-1)

    # Initial values
    # x_prev, y_prev = 330, 420
    x_prev, y_prev = 0, 0
    theta0 = 93  # hardcoded
    coordinates = []
    i = -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Plot for the translation
    fig_move = plt.figure()
    # x1 = np.linspace(-500, 500)
    # y1 = np.linspace(-500, 500)
    # line1, = plt.plot(x1, y1, 'ko-')  # so that we can update data later

    while True:
        if i == 1000:
            break

        ret, frame_1 = cap.read()
        if not ret:
            break
        i += 1
        print(f"Frame {i} out of {length}")

        # Get and enhance minimaps
        # 750, 0, 0, 1500 for minimap
        # minimap_0 = extract_enhance_minimap(frame_0, 750, 0, 1500, 0)
        # minimap_1 = extract_enhance_minimap(frame_1, 750, 0, 1500, 0)
        minimap_0 = extract_enhance_minimap(frame_0)
        minimap_1 = extract_enhance_minimap(frame_1)

        # SIFT magic
        kp1, des1, kp2, des2 = sift_keypoints_and_descriptors(minimap_0, minimap_1)
        matches = sift_flann_matches(des1, des2)
        good = sift_find_good_matches(matches)
        src_pts, dst_pts, matchesMask = sift_find_matching_keypoints(kp1, kp2, good)

        if not matchesMask:
            continue

        comparison = sift_comparison(minimap_0, kp1, minimap_1, kp2, good, matchesMask)

        # Extract rotation
        m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        theta = np.degrees(np.arctan2(-m[0, 1], m[0, 0]))

        # Rotate picture for matching
        (h, w) = minimap_1.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
        minimap_1_rotated = cv2.warpAffine(minimap_1, M, (w, h))
        cv2.imshow("ROTATE", minimap_1_rotated)

        # Match to find translation
        kp1, des1, kp2, des2 = sift_keypoints_and_descriptors(minimap_0, minimap_1_rotated)
        matches = sift_flann_matches(des1, des2)
        good = sift_find_good_matches(matches)
        src_pts, dst_pts, matchesMask = sift_find_matching_keypoints(kp1, kp2, good)
        m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        x_d = m[0, 2]
        y_d = m[1, 2]
        print(x_d, y_d)
        x, y = x_prev + x_d, y_prev + y_d
        x_prev = x
        y_prev = y

        coordinates.append([x * 10, y * 10])





        # cv2.line(minimap_0, (int(x_prev), int(y_prev)), (int(x), int(y)), (0, 255, 0), 9)
        # print(f"delta x: {x_d}, delta y: {y_d}")

        cv2.imshow("ref", frame_1)

        # cv2.imshow('Minimap 0', minimap_0)
        # cv2.imshow('Minimap 1', minimap_1)
        cv2.imshow('Comparison', comparison)

        # df = pd.DataFrame(coordinates, columns=["x", "y", "x_d", "y_d"])
        # df = df[["x", "y"]]
        # df.plot(x="x", y="y")
        # plt.plot(*zip(*coordinates))
        # plt.show()

        # update data
        # line1.set_xdata([x[0] for x in coordinates])
        # line1.set_ydata([x[1] for x in coordinates])
        colors = itertools.cycle(["r", "b", "g"])
        plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates], 'r.:')
        # redraw the canvas
        fig_move.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig_move.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig_move.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # display image with opencv or any operation you like
        cv2.imshow("plot", img)

        # Boussole
        bous = cv2.imread("../inputs/boussole.png")
        (h, w) = bous.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        theta0 -= theta
        M = cv2.getRotationMatrix2D((cX, cY), theta0, 1.0)
        rotated = cv2.warpAffine(bous, M, (w, h))
        cv2.imshow("Bousole", rotated)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

        frame_0 = frame_1

    fields = ["x", "y", "x_d", "y_d"]
    with open("../outputs/coordinates.csv", "w+") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(coordinates)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
