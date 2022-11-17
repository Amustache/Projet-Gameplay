import csv
import itertools
import json

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def extract_enhance_minimap(frame, x_0=None, x_1=None, y_0=None, y_1=None):
    """
    Take a frame, crop it, and enhance it using cv2.detailEnhance

    https://docs.opencv.org/3.4/df/dac/group__photo__render.html#ga0de660cb6f371a464a74c7b651415975

    :param frame: Image array vector
    :param x_0: !Not used
    :param x_1: !Not used
    :param y_0: !Not used
    :param y_1: !Not used
    :return: Cropped and enhanced image array vector
    """
    # EN PREMIER Y, EN DEUXIEME X
    frame = frame[750:-29, 1525:None]
    # frame = frame
    h, w, _ = frame.shape

    mask = cv2.imread('../inputs/mask_strict.png', 0)
    mask = cv2.resize(mask, (w, h))

    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    enhanced = cv2.detailEnhance(frame_masked, sigma_s=25, sigma_r=0.15)

    return enhanced


def sift_keypoints_and_descriptors(frame1, frame2):
    """
    Detects keypoints and computes the descriptors between two given frames.

    https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677

    :param frame1: Image array vector
    :param frame2: Image array vector
    :return: Keypoints and descriptors for each image array vector
    """
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    return kp1, des1, kp2, des2


def sift_flann_matches(descriptors0, descriptors1):
    """
    Compares two sets of descriptors to find matching ones.

    https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html#ab9114a6471e364ad221f89068ca21382

    https://docs.opencv.org/3.4/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89
    :param descriptors0: First set of descriptors
    :param descriptors1: Second set of descriptors
    :return: Set of matches. Each matches[i] is k or less matches for the same query descriptor.
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors0, descriptors1, k=2)

    return matches


def sift_find_good_matches(matches):
    """
    Basically returns matches that are close to each other, "good matches".

    :param matches: Set of matches
    :return: Set of good matches
    """
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good


def sift_find_matching_keypoints(kp1, kp2, good):
    """
    Helper to find homography matrix and matches mask between two sets of keypoints
    
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780

    :param kp1: Keypoints for the first image array vector
    :param kp2: Keypoints for the second image array vector
    :param good: Set of good matches
    :return: Coordinates for the found points, and the matches mask
    """
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
    """
    Generate a picture showing the comparisons

    :param frame1: Image array vector 1
    :param kp1: Corresponding keypoints
    :param frame2: Image array vector 2
    :param kp2: Corresponding keypoints
    :param good: "Good" matchings between keypoints 1 and keypoints 2
    :param matchesMask: Mask for the matches
    :return: Image array vector
    """
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    comparison = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)

    return comparison


def main():
    # cap = cv2.VideoCapture("../inputs/rotate.mp4")
    cap = cv2.VideoCapture("../inputs/video_test_extract_2.mp4")
    ret, frame_0 = cap.read()
    if not ret:
        exit(-1)

    # Initial values
    # x_prev, y_prev = 330, 420
    x_prev, y_prev = 0, 0
    # theta0 = 93  # hardcoded
    theta = 150  # hardcoded
    coordinates = []
    i = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # ["frame", "x_prev", "y_prev", "x_d", "y_d", "x_cur", "y_cur", "theta_prev", "theta_d", "theta_cur"]
    debug_list = [[i, x_prev, y_prev, 0, 0, x_prev, y_prev, theta, 0, theta]]

    # Plot for the translation
    fig_move = plt.figure()

    # While we have frames
    while True:
        ret, frame_1 = cap.read()
        if not ret:
            break
        i += 1
        print(f"Frame {i} out of {length}")

        # Here we jump five frames each time so that we have more room for comparison
        if i % 5 != 0:
            continue

        # Get and enhance minimaps
        # 750, -29, 1525, 0 for minimap
        minimap_0 = extract_enhance_minimap(frame_0)
        minimap_1 = extract_enhance_minimap(frame_1)

        # SIFT magic
        kp1, des1, kp2, des2 = sift_keypoints_and_descriptors(minimap_0, minimap_1)
        matches = sift_flann_matches(des1, des2)
        good = sift_find_good_matches(matches)
        src_pts, dst_pts, matchesMask = sift_find_matching_keypoints(kp1, kp2, good)

        # If we have no matches, we cannot compare
        if not matchesMask:
            continue

        comparison = sift_comparison(minimap_0, kp1, minimap_1, kp2, good, matchesMask)

        # Extract rotation
        m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        theta_current = np.degrees(np.arctan2(-m[0, 1], m[0, 0]))  # A vérifier

        # Rotate picture for matching
        (h, w) = minimap_0.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
        minimap_0_rotated = cv2.warpAffine(minimap_0, M, (w, h))

        (h, w) = minimap_1.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), theta + theta_current, 1.0)
        minimap_1_rotated = cv2.warpAffine(minimap_1, M, (w, h))

        # Match to find translation
        kp1, des1, kp2, des2 = sift_keypoints_and_descriptors(minimap_0_rotated, minimap_1_rotated)
        matches = sift_flann_matches(des1, des2)
        good = sift_find_good_matches(matches)
        src_pts, dst_pts, matchesMask = sift_find_matching_keypoints(kp1, kp2, good)
        m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Discard "big moves"
        THREESHOLD_LOW = 0.01
        THREESHOLD_HIGH = 1

        x_d = m[0, 2]
        if abs(x_d) < THREESHOLD_LOW or abs(x_d) > THREESHOLD_HIGH:
            x_d = 0.0
        y_d = m[1, 2]
        if abs(y_d) < THREESHOLD_LOW or abs(y_d) > THREESHOLD_HIGH:
            y_d = 0.0

        x, y = x_prev + x_d, y_prev + y_d
        x_prev = x
        y_prev = y

        coordinates.append([x, y])
        theta += theta_current
        frame_0 = frame_1

        debug_list.append([i, x_prev - x_d, y_prev - y_d, x_d, y_d, x_prev, y_prev, theta - theta_current, theta_current, theta])

        # Show things
        ref = cv2.resize(frame_1, (int(frame_1.shape[1] * 50 / 100), int(frame_1.shape[0] * 50 / 100)))
        ref = cv2.putText(ref, f"d_x: {x_d}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        ref = cv2.putText(ref, f"d_y: {y_d}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # update data
        plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates], 'r.:')
        # redraw the canvas
        fig_move.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig_move.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig_move.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # display image with opencv or any operation you like

        # Boussole
        bous = cv2.imread("../inputs/boussole.png")
        (h, w) = bous.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -theta, 1.0)
        rotated = cv2.warpAffine(bous, M, (w, h))

        cv2.imshow("Reference", ref)
        cv2.imshow('Minimap 0', minimap_0)
        cv2.imshow('Minimap 0 rotated', minimap_0_rotated)
        cv2.imshow('Minimap 1', minimap_1)
        cv2.imshow('Minimap 1 rotated', minimap_1_rotated)
        cv2.imshow('Comparison', comparison)
        cv2.imshow("plot", img)
        cv2.imshow("Bousole", rotated)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # When we're done, extract to a file
    fields = ["frame", "x_prev", "y_prev", "x_d", "y_d", "x_cur", "y_cur", "theta_prev", "theta_d", "theta_cur"]
    df = pd.DataFrame(debug_list, columns=fields)
    df.to_csv("../outputs/coordinates.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
