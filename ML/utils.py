import json
import sys


from inc.KeyslogReader import KeyslogReader
import cv2
import pandas


def anotate():
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        sys.exit()

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)
    position = (50, 50)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    video_in = cv2.VideoCapture(sys.argv[2])
    frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_in.get(cv2.CAP_PROP_FPS)
    total_frames = video_in.get(cv2.CAP_PROP_FRAME_COUNT)

    video_out = cv2.VideoWriter("out.mp4", fourcc, fps, (frame_width, frame_height))
    keys = KeyslogReader(sys.argv[3])

    current_frame = 0
    valid, frame = video_in.read()
    while valid:

        if current_frame % 100 == 0:
            print(f"Working... ({100*current_frame/total_frames:.1f}%)")

        txt = str(keys[current_frame])
        cv2.putText(frame, txt, position, font, 1, color)
        video_out.write(frame)
        valid, frame = video_in.read()
        current_frame += 1

    print("Finished.")


def csvToJs(file_path, dataName):
    output = []
    f = open(file_path, "r")
    f.readline()
    for line in f:
        if line.strip() != "":
            line_split = line.strip().split(",")
            line_split[0] = int(line_split[0])
            output.append(line_split)

    f_output = open(file_path.split(".")[0] + ".js", "w")
    f_output.write("const " + dataName + "=")
    f_output.write(json.dumps(output))

    f.close()
    f_output.close()


def main():
    if len(sys.argv) < 2:
        print("No action specified !")
    elif sys.argv[1] == "anotate":
        anotate()
    elif sys.argv[1] == "csvToJs":
        csvToJs(sys.argv[2], sys.argv[3])
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
