import random
import time

import cv2
import pandas as pd

from helpers import *


def main(show=False, verbatim=False):
    # test_rotation(show, verbatim)
    # test_translation(show, verbatim, 200)
    test_translation_and_rotation(show, verbatim, 200)


def add_gaussian_noise(img):
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')

    img_gauss = cv2.add(img, gauss)

    return img_gauss

def get_random_translation():
    return np.float32([
        [1, 0, random.randint(-10, 10)],
        [0, 1, random.randint(-10, 10)]
    ])


def test_translation_and_rotation(show=False, verbatim=False, iters=50):
    base = cv2.imread("../inputs/dummy_base.png")
    mask = cv2.imread('../inputs/mask_strict.png', 0)

    errors = list()

    for i in range(0, iters):

        if verbatim:
            print(f"== current iteration: {1+i}/{iters} ==")
        translate_first = random.randint(1, 6) > 3
        if verbatim:
            print(f"translate first: {translate_first}")

        theta_truth = round(random.uniform(-180, 180), 1)
        M = get_random_translation()
        x_truth, y_truth = M[0, 2], M[1, 2]
        if verbatim:
            print(f"truth: theta: {theta_truth}, x: {x_truth}, y: {y_truth}")

        if translate_first:
            cur = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]))
            cur = straighten_img(cur, theta_truth)
        else:
            cur = straighten_img(base, theta_truth)
            cur = cv2.warpAffine(cur, M, (cur.shape[1], cur.shape[0]))

        cur = add_gaussian_noise(cur)

        if show:
            cv2.imshow("base", base)
            cv2.imshow("cur", cur)

        minimap_0 = extract_enhance_minimap(base, mask)
        minimap_1 = extract_enhance_minimap(cur, mask)
        if show:
            cv2.imshow("minimap_0", minimap_0)
            cv2.imshow("minimap_1", minimap_1)

        # SIFT magic
        src_pts, dst_pts, matchesMask, comparison = sift(minimap_0, minimap_1, True)
        assert matchesMask is not None, "matchesMask is None"

        # Extract rotation
        try:
            scale, theta_relative, _, _ = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue
        if verbatim:
            print(f"theta found: {theta_relative}, expected: {-theta_truth}")
        # assert theta_relative == -theta_truth, f"found rotation is not correct: {theta_relative} should be {-theta_truth}"

        # Rotate picture for matching
        minimap_1_rotated = straighten_img(minimap_1, theta_relative)
        if show:
            cv2.imshow("minimap_1_rotated", minimap_1_rotated)
        test_x, test_y = cross_diff(minimap_0, minimap_1_rotated)
        if verbatim:
            print(f"! found rotation shift: {test_x, test_y}")

        # Match to find translation
        src_pts, dst_pts, _ = sift(minimap_0, minimap_1_rotated)
        try:
            _, _, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue
        if verbatim:
            print(f"! found x_d: {x_d}, found y_d: {y_d}")
        errors.append([abs(theta_truth + theta_relative), abs(x_truth - x_d), abs(y_truth - y_d)])

        if show:
            k = cv2.waitKey(1000) & 0xFF
            if k == 27:
                break
            cv2.destroyAllWindows()

    df = pd.DataFrame(errors, columns=["theta", "x_d", "y_d"])
    print(df.mean())


def test_translation(show=False, verbatim=False, iters=50):
    base = cv2.imread("../inputs/dummy_base.png")
    mask = cv2.imread('../inputs/mask_strict.png', 0)

    neutral = np.float32([
        [1, 0, 0],
        [0, 1, 0]
    ])
    temp = cv2.warpAffine(base, neutral, (base.shape[1], base.shape[0]))
    test = cross_diff(base, temp)
    assert test == (0, 0)

    for i in range(0, iters):
        if verbatim:
            print(f"== current translation: {1+i}/{iters} ==")
        M = get_random_translation()
        x_truth, y_truth = M[0, 2], M[1, 2]
        if verbatim:
            print(f"truth: {x_truth, y_truth}")

        cur = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]))
        test = cross_diff(base, cur)
        if verbatim:
            print(f"found cross: {test}")
        assert test == (x_truth, y_truth), "translation is not correct"

        minimap_0 = extract_enhance_minimap(base, mask)
        minimap_1 = extract_enhance_minimap(cur, mask)
        if show:
            cv2.imshow("minimap_0", minimap_0)
            cv2.imshow("minimap_1", minimap_1)

        # SIFT magic
        src_pts, dst_pts, matchesMask, comparison = sift(minimap_0, minimap_1, True)
        assert matchesMask is not None, "matchesMask is None"

        try:
            _, _, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue
        if verbatim:
            print(f"found x_d: {x_d}, found y_d: {y_d}")
        assert x_d == x_truth and y_d == y_truth, "found values are not correct"

        # Translate image back
        M_back = np.float32([
            [1, 0, -x_d],
            [0, 1, -y_d]
        ])
        minimap_1_translated = cv2.warpAffine(minimap_1, M_back, (minimap_1.shape[1], minimap_1.shape[0]))
        test_x, test_y = cross_diff(minimap_0, minimap_1_translated)
        if verbatim:
            print(f"final cross: {test}")
        assert abs(test_x) <= 1 and abs(test_y) <= 1, "translation is not correct"

        if show:
            k = cv2.waitKey(1000) & 0xFF
            if k == 27:
                break
            cv2.destroyAllWindows()


def test_rotation(show=False, verbatim=False):
    base = cv2.imread("../inputs/dummy_base.png")
    mask = cv2.imread('../inputs/mask_strict.png', 0)

    temp = straighten_img(base, 0)
    test = cross_diff(base, temp)
    assert test == (0, 0)

    for i, rot in enumerate(range(-180, 180, 10)):
        if verbatim:
            print(f"== current rotation: {rot} ({1+i}/{len(range(-180, 180, 10))}) ==")

        # Test rotation
        cur = straighten_img(base, rot)  # Anti-clockwise
        test = cross_diff(base, straighten_img(cur, -rot))
        if show:
            cv2.imshow("base", base)
            cv2.imshow("cur", cur)
        if verbatim:
            print(f"Initial rotation shift: {test}")
        assert test == (0, 0), f"rotation is not correct: {test}"

        minimap_0 = extract_enhance_minimap(base, mask)
        minimap_1 = extract_enhance_minimap(cur, mask)
        if show:
            cv2.imshow("minimap_0", minimap_0)
            cv2.imshow("minimap_1", minimap_1)

        # SIFT magic
        src_pts, dst_pts, matchesMask, comparison = sift(minimap_0, minimap_1, True)
        assert matchesMask is not None, "matchesMask is None"

        # Extract rotation
        try:
            scale, theta_relative, _, _ = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue
        if verbatim:
            print(f"theta found: {theta_relative}, expected: {-rot}")
        assert abs(theta_relative + rot) < 0.2, f"found rotation is not correct: {theta_relative} should be {-rot}"

        # Rotate picture for matching
        minimap_1_rotated = straighten_img(minimap_1, theta_relative)
        if show:
            cv2.imshow("minimap_1_rotated", minimap_1_rotated)
        test_x, test_y = cross_diff(minimap_0, minimap_1_rotated)
        if verbatim:
            print(f"found rotation shift: {test_x, test_y}")
        assert abs(test_x) <= 1 and abs(test_y) <= 1, f"rotation is not correct: {test_x, test_y}"

        # Match to find translation
        src_pts, dst_pts, _ = sift(minimap_0, minimap_1_rotated)
        try:
            _, _, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue
        if verbatim:
            print(f"found x_d: {x_d}, found y_d: {y_d}")
        assert x_d == 0 and y_d == 0

        if show:
            k = cv2.waitKey(1000) & 0xFF
            if k == 27:
                break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main(show=False, verbatim=True)
