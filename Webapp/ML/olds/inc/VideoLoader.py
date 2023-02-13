import cv2
import numpy as np
import torch


class VideoLoader:
    def __init__(self, video_path, dimensions, start_frame=0, end_frame=0, fps=30, preload=False):
        self.video = cv2.VideoCapture(video_path)
        self.output_width, self.output_height = dimensions
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.preload = preload

        self.fps_step = int(self.video.get(cv2.CAP_PROP_FPS) / fps)

        self.cache = {}
        self.previous_id = 0
        self.max_cache_size = 20

        if self.preload:
            self.data = []
            self.data_time = []
            self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if end_frame == 0:
                end_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_rgb = []
            frame_grey = []
            print(f"Loading video ({end_frame-start_frame} frames)...")
            valid = True
            for i in range(start_frame, end_frame, self.fps_step):
                if i % 1000 == 0:
                    print(f"Frame {i}...")
                for j in range(self.fps_step):
                    valid, frame_rgb = self.video.read()
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
        if self.end_frame > self.start_frame:
            return self.end_frame - self.start_frame
        else:
            return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps_step - self.start_frame)

    def __getitem__(self, idx):
        f0 = self.getFrame(idx)
        f1 = self.getFrame(idx + 1)
        f2 = self.getFrame(idx + 2)
        f3 = self.getFrame(idx + 3)
        f4 = self.getFrame(idx + 4)
        f5 = self.getFrame(idx + 5)
        f6 = self.getFrame(idx + 6)
        f7 = self.getFrame(idx + 7)
        f8 = self.getFrame(idx + 8)
        f9 = self.getFrame(idx + 9)
        f10 = self.getFrame(idx + 10)
        f11 = self.getFrame(idx + 11)
        f12 = self.getFrame(idx + 12)
        f13 = self.getFrame(idx + 13)
        f14 = self.getFrame(idx + 14)
        f15 = self.getFrame(idx + 15)

        fd0 = f1[1] - f0[1]
        fd1 = f2[1] - f1[1]
        fd2 = f3[1] - f2[1]
        fd3 = f4[1] - f3[1]
        fd4 = f5[1] - f4[1]
        fd5 = f7[1] - f6[1]
        fd6 = f9[1] - f8[1]
        fd7 = f11[1] - f10[1]
        fd8 = f13[1] - f12[1]
        fd9 = f15[1] - f14[1]
        return f0[1], torch.cat((fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7, fd8, fd9), dim=0)

    def getFrame(self, id):
        if self.preload:
            if id < len(self.data):
                return self.data[id], self.data_time[id]
            else:
                return self.getFrame(len(self.data) - 1)

        elif id in self.cache.keys():
            return self.cache[id]

        else:
            if id != self.previous_id + 1:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, (id - 1) * self.fps_step + self.start_frame)

            for i in range(self.fps_step):
                valid, frame = self.video.read()

            self.previous_id = id
            if valid:
                result = self._process_frame(frame)

                self.cache[id] = result
                if len(self.cache) > self.max_cache_size:
                    del self.cache[min(self.cache.keys())]

                return result
            else:
                return self.getFrame(self.__len__() - 1)

    def getFrameStep(self):
        return self.fps_step

    def _process_frame(self, frame):
        frame = cv2.resize(frame, (self.output_width, self.output_height))
        frame_rgb = torch.permute(torch.FloatTensor(frame / 255), (2, 0, 1))

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grey = torch.FloatTensor(frame_grey[None, :, :] / 255)

        return frame_rgb, frame_grey
