import cv2
from helpers import *


def main():
    base = cv2.imread("../inputs/dummy_base.png")
    rotate_left = cv2.imread("../inputs/dummy_rotate_left.png")
    rotate_right = cv2.imread("../inputs/dummy_rotate_right.png")
    translate_then_rotate = cv2.imread("../inputs/dummy_translate_then_rotate.png")
    translate_x = cv2.imread("../inputs/dummy_translate_x.png")
    translate_y = cv2.imread("../inputs/dummy_translate_y.png")
    translate_xy = cv2.imread("../inputs/dummy_translate_xy.png")

    mask = cv2.imread('../inputs/mask_strict.png', 0)

    while True:
        choice = rotate_left

        minimap_0 = extract_enhance_minimap(base, mask)
        minimap_1 = extract_enhance_minimap(choice, mask)

        src_pts, dst_pts, _ = sift(minimap_0, minimap_1)

        scale, theta, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        print(f"Before: scale: {scale}; theta: {theta}; x_d: {x_d}; y_d: {y_d}")

        # Redresser
        minimap_1 = straighten_img(minimap_1, theta, scale)

        # Vérifier
        src_pts, dst_pts, _ = sift(minimap_0, minimap_1)

        scale, theta, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        print(f"After: scale: {scale}; theta: {theta}; x_d: {x_d}; y_d: {y_d}")

        cv2.imshow("base", base)
        cv2.imshow("choice", choice)
        cv2.imshow("minimap_0", minimap_0)
        cv2.imshow("minimap_1", minimap_1)

        # Get cross diff
        cross_x_d, cross_y_d = cross_diff(minimap_0, minimap_1)
        print(f"cross diff: {cross_x_d, cross_y_d}")

        # Mean of the two
        print(f"x_d: {(x_d + cross_x_d) // 2}; y_d: {(y_d + cross_y_d) // 2}")

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
