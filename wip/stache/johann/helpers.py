import cv2
import numpy as np
import scipy.signal


######## JP TEST FUNCTIONS ############


def enhance_fg(img):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gimg, 110, 255, cv2.THRESH_BINARY)

    boxed = cv2.boxFilter(thresh1, -1, (40, 40), normalize=False)
    alpha = cv2.blur(boxed, (11, 11))
    alpha = alpha.reshape(*alpha.shape, 1).astype("float32") / 255

    fg = (img * alpha).astype("uint8")
    composed = cv2.addWeighted(img, 0.2, fg, 0.8, 0)
    return composed


def filter_kp(kp, img, radius=115):
    center = np.float32(img.shape[1::-1]) / 2
    return [pt for pt in kp if np.linalg.norm(pt.pt - center) < radius]


def kp_description(frame, filter_radius=False, use_binary=False):
    descriptor = cv2.BRISK.create() if use_binary else cv2.KAZE.create(upright=True)

    detector = cv2.KAZE.create(upright=True)

    kp1 = detector.detect(frame)

    if filter_radius:
        kp1 = filter_kp(kp1, frame)

    kp1, des1 = descriptor.compute(frame, kp1)

    return kp1, des1


def kp_matching(des1, des2, ann=False, k=2, use_binary=False):
    if ann:  # WARNING Not finished
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1  # 12  # 20
        )  # 2
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors0, descriptors1, k=k)
    else:  # slower but not approximate
        bf = cv2.BFMatcher(cv2.NORM_HAMMING if use_binary else cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=k)
    return matches


def filter_with_template(template_des, kp1, des1, k=8, use_binary=False):
    matches = kp_matching(template_des, des1, k=k, use_binary=use_binary)
    kp = [kp1[m.trainIdx] for m in matches[0]]  # can do better than 0
    des = [des1[m.trainIdx] for m in matches[0]]  # can do better than 0
    return kp, np.vstack(des)


def filter_matches(matches, ratio_test=False):
    if ratio_test:
        good = []
        for m, n in matches:  # Lowe's SIFT value may not be relevant for binary descriptors
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return good
    else:
        return [m[0] for m in matches]


def draw_matches(frame1, kp1, frame2, kp2, good, matchesMask=None):
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    comparison = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)
    return comparison


###################


ERROR_THRESHOLD = 0  # Number of digits kept


def straighten_img(img, theta=0, scale=1):
    if abs(theta) > 180:
        print(f"Theta has strange value {theta}")
        # raise ValueError("-180 <= theta <= 180")
    if theta == 180:
        theta = -180

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), theta, scale)
    img = cv2.warpAffine(img, M, (w, h))

    return img


def get_affine_matrix(src_pts, dst_pts, threshold=ERROR_THRESHOLD):
    m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    scale = round(np.sign(m[0, 0]) * np.sqrt(m[0, 0] ** 2 + m[0, 1] ** 2), threshold)
    theta = round(np.degrees(np.arctan2(-m[0, 1], m[0, 0])), 1 + threshold)
    x_d = round(m[0, 2], threshold)
    y_d = round(m[1, 2], threshold)

    return scale, theta, x_d, y_d


def sift(frame1, frame2, comparison=False):
    kp1, des1, kp2, des2 = sift_keypoints_and_descriptors(frame1, frame2)
    matches = sift_flann_matches(des1, des2)
    good = sift_find_good_matches(matches)
    src_pts, dst_pts, matchesMask = sift_find_matching_keypoints(kp1, kp2, good)

    if comparison:
        return (
            src_pts,
            dst_pts,
            matchesMask,
            sift_comparison(frame1, kp1, frame2, kp2, good, matchesMask),
        )
    return src_pts, dst_pts, matchesMask


def cross_diff(im1, im2):
    corr_img = cross_image(im1, im1)
    corr1 = np.unravel_index(np.argmax(corr_img), corr_img.shape)

    corr_img = cross_image(im1, im2)
    corr2 = np.unravel_index(np.argmax(corr_img), corr_img.shape)

    return corr1[1] - corr2[1], corr1[0] - corr2[0]


def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = np.sum(im1.astype("float"), axis=2)
    im2_gray = np.sum(im2.astype("float"), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode="same")


def ec(number, threshold=0.001):
    return number if abs(number) > threshold else 0


def extract_enhance_minimap(frame, mask=None, x_0=None, x_1=None, y_0=None, y_1=None):
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
    h, w, _ = frame.shape
    if not x_0:
        x_0 = 0
    if not x_1:
        x_1 = h
    if not y_0:
        y_0 = 0
    if not y_1:
        y_1 = w
    frame = frame[x_0:x_1, y_0:y_1]
    h, w, _ = frame.shape

    if mask is not None and mask.any():
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

    algo = cv2.SIFT_create()

    kp1, des1 = algo.detectAndCompute(frame1, None)
    kp2, des2 = algo.detectAndCompute(frame2, None)

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

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors0, descriptors1, k=2)
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


def sift_find_matching_keypoints(kp1, kp2, good, min_match_count=5):
    """
    Helper to find homography matrix and matches mask between two sets of keypoints

    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780

    :param kp1: Keypoints for the first image array vector
    :param kp2: Keypoints for the second image array vector
    :param good: Set of good matches
    :return: Coordinates for the found points, and the matches mask
    """
    if len(good) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
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
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    comparison = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)

    return comparison
