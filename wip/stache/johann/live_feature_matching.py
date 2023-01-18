from helpers import *
from matplotlib import pyplot as plt
import pandas as pd


### JP PARAMS FOR TEST ###

USE_BINARY_KP = False
FORCE_ENHANCE_FG = True
TEMPLATE_FILTER = True
TEMPLATE_FILTER_K = 15


def main():
    # cap = cv2.VideoCapture("../inputs/rotate.mp4")
    cap = cv2.VideoCapture("../inputs/video_test_extract_2.mp4")

    fourcc = cv2.VideoWriter.fourcc(*"MP4V")
    vid_writer = cv2.VideoWriter("../outputs/kp_path.mp4", fourcc, 20.0, (960, 540 + 322))

    # cap = cv2.VideoCapture("../inputs/extract_pokemon.mp4")
    ret, frame_0 = cap.read()
    if not ret:
        exit(-1)

    mask = cv2.imread("../inputs/mask_strict.png", 0)

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

    ## JP
    loc = np.float32([0, 0])

    if TEMPLATE_FILTER:
        template = cv2.imread("../inputs/cross.png")
        _, template_des = kp_description(template, use_binary=USE_BINARY_KP)

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

        if i > 2500:
            break

        # Get and enhance minimaps
        # 750, -29, 1525, 0 for minimap zelda
        minimap_0 = extract_enhance_minimap(frame_0, mask, 750, -29, 1525)
        minimap_1 = extract_enhance_minimap(frame_1, mask, 750, -29, 1525)

        failed_attempts = 1 if FORCE_ENHANCE_FG else 0

        while failed_attempts < 2:

            if failed_attempts == 1:
                minimap_0 = enhance_fg(minimap_0)
                minimap_1 = enhance_fg(minimap_1)

            ## NEW METHOD
            kp1, des1 = kp_description(minimap_0, use_binary=USE_BINARY_KP)
            kp2, des2 = kp_description(minimap_1, use_binary=USE_BINARY_KP)

            if TEMPLATE_FILTER:
                kp1, des1 = filter_with_template(
                    template_des, kp1, des1, k=TEMPLATE_FILTER_K, use_binary=USE_BINARY_KP
                )
                kp2, des2 = filter_with_template(
                    template_des, kp2, des2, k=TEMPLATE_FILTER_K, use_binary=USE_BINARY_KP
                )

            matches = kp_matching(des1, des2, use_binary=USE_BINARY_KP)
            matches = filter_matches(matches, ratio_test=not TEMPLATE_FILTER)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            comparison = draw_matches(minimap_0, kp1, minimap_1, kp2, matches)

            # SIFT magic
            # src_pts, dst_pts, matchesMask, comparison = sift(minimap_0, minimap_1, True)

            # If we have no matches, we cannot compare
            # if not matchesMask:
            #     print(f"Error: No matchesMask")
            #     continue

            ## JP
            m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

            # Get perspective matrix (for easier single point manipulation)
            Mp = np.zeros([3, 3])
            Mp[:2, :3] = m
            Mp[2, 2] = 1

            # Get center for src mmap (it needs to be VERY precise, otherwise it will cause drifts)
            center = np.float32(minimap_0.shape[1::-1]) / 2
            ## HARDCODED
            center = np.float32([200.5, 151])

            # Map center (cursor) using affine transform disguised in perspective matrix for convenience
            new_center = cv2.perspectiveTransform(center.reshape(-1, 1, 2), Mp)

            # Compute delta
            delta = new_center - center

            if np.linalg.norm(delta) > 1:
                print(f"ERROR #{failed_attempts+1} {np.linalg.norm(delta)}")
                delta = np.zeros([2, 1])
                failed_attempts += 1
            else:
                break

        frame_0 = frame_1

        # Find new location on grid
        new_loc = loc + delta.ravel()
        print(np.linalg.norm(delta), loc)
        loc = new_loc

        # # Extract rotation
        # try:
        #     scale, theta_relative, _, _ = get_affine_matrix(src_pts, dst_pts)

        #     m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        #     print(m)
        # except cv2.error:
        #     print(f"Error: No affine matrix found")
        #     continue

        # # Rotate picture for matching
        # minimap_0_rotated = straighten_img(minimap_0, theta_for_north, scale)
        # minimap_1_rotated = straighten_img(minimap_1, theta_for_north + theta_relative, scale)

        # # Match to find translation
        # src_pts, dst_pts, _ = sift(minimap_0_rotated, minimap_1_rotated)
        # try:
        #     _, _, x_d, y_d = get_affine_matrix(src_pts, dst_pts)
        # except cv2.error:
        #     print(f"Error: No affine matrix found")
        #     continue
        # print(x_d, y_d)

        ### Map loc to local variables

        x_prev, y_prev = loc[0], loc[1]
        x_d, y_d = delta.ravel()[0], delta.ravel()[1]

        x, y = x_prev + x_d, y_prev + y_d
        x_prev = x
        y_prev = y

        coordinates.append([x, y])
        # theta_for_north += theta_relative

        # debug_list.append([i, x_prev - x_d, y_prev - y_d, x_d, y_d, x_prev, y_prev, theta_for_north - theta_relative, theta_relative, theta_for_north])

        # Show things
        ref = cv2.resize(
            frame_1, (int(frame_1.shape[1] * 50 / 100), int(frame_1.shape[0] * 50 / 100))
        )
        ref = cv2.putText(
            ref, f"d_x: {x_d}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
        )
        ref = cv2.putText(
            ref, f"d_y: {y_d}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
        )

        # update data
        plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates], "r.:")
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
        # bous = cv2.imread("../inputs/boussole.png")
        # (h, w) = bous.shape[:2]
        # (cX, cY) = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D((cX, cY), -theta_for_north, 1.0)
        # rotated = cv2.warpAffine(bous, M, (w, h))

        # cv2.imshow("Reference", ref)
        # cv2.imshow('Minimap 0', minimap_0)
        # cv2.imshow('Minimap 0 rotated', minimap_0_rotated)
        # cv2.imshow('Minimap 1', minimap_1)
        # cv2.imshow('Minimap 1 rotated', minimap_1_rotated)
        # cv2.imshow('Comparison', comparison)
        # cv2.imshow("plot", img)
        # cv2.imshow("Bousole", rotated)

        render = np.zeros([540 + 322, 960, 3])
        render[:540, :960, :] = ref
        render[540:742, :530] = cv2.resize(comparison, (530, 202))
        render[540:, 530:] = cv2.resize(img, (430, 322))

        ## JP
        # cv2.imwrite("../inputs/mmap0.png", minimap_0)
        # cv2.imwrite("../inputs/mmap1.png", minimap_1)
        vid_writer.write(render.astype("uint8"))

        # k = cv2.waitKey() & 0xFF
        # if k == 27:
        #    break

    # When we're done, extract to a file
    fields = [
        "frame",
        "x_prev",
        "y_prev",
        "x_d",
        "y_d",
        "x_cur",
        "y_cur",
        "theta_prev",
        "theta_d",
        "theta_cur",
    ]
    df = pd.DataFrame(debug_list, columns=fields)
    df.to_csv("../outputs/coordinates.csv", index=False)

    cap.release()
    vid_writer.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
