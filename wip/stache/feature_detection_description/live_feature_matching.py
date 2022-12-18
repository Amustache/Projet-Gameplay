import pandas as pd
from matplotlib import pyplot as plt
from helpers import *


def main():
    # cap = cv2.VideoCapture("../inputs/rotate.mp4")
    # cap = cv2.VideoCapture("../inputs/video_test_extract_2.mp4")
    cap = cv2.VideoCapture("../inputs/extract_pokemon.mp4")
    ret, frame_0 = cap.read()
    if not ret:
        exit(-1)

    mask = cv2.imread('../inputs/mask_strict.png', 0)

    # Initial values
    x_prev, y_prev = 0, 0
    theta_for_north = 160  # hardcoded
    i = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    coordinates = []
    # ["frame", "x_prev", "y_prev", "x_d", "y_d", "x_cur", "y_cur", "theta_prev", "theta_d", "theta_cur"]
    debug_list = [[i, x_prev, y_prev, 0, 0, x_prev, y_prev, theta_for_north, 0, theta_for_north]]

    # Plot for the translation
    fig_move = plt.figure()

    # While we have frames
    while True:
        ret, frame_1 = cap.read()
        if not ret:
            break
        i += 1
        # print(f"Frame {i} out of {total_frames}")

        # Here we jump five frames each time so that we have more room for comparison
        if i % 5 != 0:
            continue

        # Get and enhance minimaps
        # 750, -29, 1525, 0 for minimap zelda
        minimap_0 = extract_enhance_minimap(frame_0, mask, 750, -29, 1525)
        minimap_1 = extract_enhance_minimap(frame_1, mask, 750, -29, 1525)

        # SIFT magic
        src_pts, dst_pts, matchesMask, comparison = sift(minimap_0, minimap_1, True)

        # If we have no matches, we cannot compare
        if not matchesMask:
            print(f"Error: No matchesMask")
            continue

        # Extract rotation
        try:
            scale, theta_relative, _, _ = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue

        # Rotate picture for matching
        minimap_0_rotated = straighten_img(minimap_0, theta_for_north, scale)
        minimap_1_rotated = straighten_img(minimap_1, theta_for_north + theta_relative, scale)

        # Match to find translation
        src_pts, dst_pts, _ = sift(minimap_0_rotated, minimap_1_rotated)
        try:
            _, _, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        except cv2.error:
            print(f"Error: No affine matrix found")
            continue
        print(x_d, y_d)

        x, y = x_prev + x_d, y_prev + y_d
        x_prev = x
        y_prev = y

        coordinates.append([x, y])
        theta_for_north += theta_relative
        frame_0 = frame_1

        # debug_list.append([i, x_prev - x_d, y_prev - y_d, x_d, y_d, x_prev, y_prev, theta_for_north - theta_relative, theta_relative, theta_for_north])

        # Show things
        ref = cv2.resize(frame_1, (int(frame_1.shape[1] * 50 / 100), int(frame_1.shape[0] * 50 / 100)))
        ref = cv2.putText(ref, f"d_x: {x_d}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        ref = cv2.putText(ref, f"d_y: {y_d}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # update data
        plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates], 'r.:')
        # redraw the canvas
        fig_move.canvas.draw()
        # convert canvas to image
        img = np.frombuffer(fig_move.canvas.tostring_rgb(), dtype=np.uint8)
        testshape = fig_move.canvas.get_width_height()[::-1] + (3,)
        # print(img.shape)
        img = img.reshape(fig_move.canvas.get_width_height()[::-1] + (3,))
        # print(testshape)
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # display image with opencv or any operation you like

        # Boussole
        bous = cv2.imread("../inputs/boussole.png")
        (h, w) = bous.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -theta_for_north, 1.0)
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
