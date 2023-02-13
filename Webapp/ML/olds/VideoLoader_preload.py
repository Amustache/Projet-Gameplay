import cv2
import numpy as np
import torch


class VideoLoader:
    def __init__(self, video_path, dimensions, start_frame=0, end_frame=0, fps=30):
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        self.output_width, self.output_height = dimensions
        self.fps_step = int(video.get(cv2.CAP_PROP_FPS) / fps)

        self.data = []
        self.data_time = []

        if end_frame == 0:
            end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_rgb = []
        frame_grey = []

        print(f"Loading video ({end_frame-start_frame} frames)...")

        valid = True
        for i in range(start_frame, end_frame, self.fps_step):
            if i % 1000 == 0:
                print(f"Frame {i}...")

            for j in range(self.fps_step):
                valid, frame_rgb = video.read()

            if valid:
                frame_rgb, frame_grey = self._process_frame(frame_rgb)
                if len(frame_rgb) != 0:
                    self.data.append(frame_rgb)
                    self.data_time.append(frame_grey)

        for i in range(5):
            self.data_time.append(frame_grey)

        print(
            f"Video loaded. Nb of output frames : {int((end_frame-start_frame)/self.fps_step)} \n"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f1 = self.data_time[idx + 1] - self.data_time[idx]
        f2 = self.data_time[idx + 2] - self.data_time[idx + 1]
        f3 = self.data_time[idx + 3] - self.data_time[idx + 2]
        f4 = self.data_time[idx + 4] - self.data_time[idx + 3]
        f5 = self.data_time[idx + 5] - self.data_time[idx + 4]
        return self.data[idx], torch.cat((f1, f2, f3, f4, f5), dim=0)

    def getFrameStep(self):
        return self.fps_step

    def _process_frame(self, frame):
        frame = cv2.resize(frame, (self.output_width, self.output_height))
        frame_rgb = torch.permute(torch.FloatTensor(frame / 255), (2, 0, 1))

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grey = torch.FloatTensor(frame_grey[None, :, :] / 255)

        return frame_rgb, frame_grey
