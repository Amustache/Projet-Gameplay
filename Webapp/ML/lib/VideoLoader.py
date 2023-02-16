import torch
import torchvision.transforms as T
import cv2
import numpy as np

class VideoLoader:
    def __init__(self, video_path, dimensions, start_frame=0, end_frame=0, fps=30, preload=False):
        self.video = cv2.VideoCapture(video_path)
        self.output_width, self.output_height = dimensions
        self.start_frame = start_frame
        self.preload = preload

        self.fps_step  = int(self.video.get(cv2.CAP_PROP_FPS)/fps)

        self.cache = {}
        self.cache_frames = {}
        self.cache_flow = {}

        self.previous_id = 0
        self.max_cache_size = 64

        if self.preload :
            current = None
            previous = None
            self.time_data = []
            self.data = []
            window = []
            self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if end_frame == 0:
                end_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Loading video ({end_frame-start_frame} frames)...")
            valid = True
            for i in range(start_frame, end_frame, self.fps_step):
                if i % 1000 == 0 : print(f"Frame {i}...")
                valid, current = self._get_frame(i)
                if valid :
                    self.data.append( torch.FloatTensor(current/255) )
                    if previous is not None :
                        window.append(self._get_motionFlow(i, previous, current))
                        if len(window) == 4:
                            res_img = self._motionFlow_to_rgb(previous, window)
                            self.time_data.append( torch.FloatTensor(res_img/255) )
                            window.pop(0)

                    previous = current

            self.time_data.append( torch.FloatTensor(res_img/255) )
            self.time_data.append( torch.FloatTensor(res_img/255) )
            self.time_data.append( torch.FloatTensor(res_img/255) )
            self.time_data.append( torch.FloatTensor(res_img/255) )
            print(f"Video loaded. Nb of output frames : {int((end_frame-start_frame)/self.fps_step)} \n")


    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)/self.fps_step - self.start_frame)


    def __getitem__(self, idx):
        f0  = self.getFrame(idx)
        f1  = self.getFrame(idx+1)
        f2  = self.getFrame(idx+2)
        f3  = self.getFrame(idx+3)
        f4  = self.getFrame(idx+4)
        f5  = self.getFrame(idx+5)
        f6  = self.getFrame(idx+6)
        f7  = self.getFrame(idx+7)
        f8  = self.getFrame(idx+8)
        f9  = self.getFrame(idx+9)
        f10 = self.getFrame(idx+10)
        f11 = self.getFrame(idx+11)
        f12 = self.getFrame(idx+12)
        f13 = self.getFrame(idx+13)
        f14 = self.getFrame(idx+14)
        f15 = self.getFrame(idx+15)
        
        fd0 = f1[0]-f0[0]
        fd1 = f2[0]-f1[0]
        fd2 = f3[0]-f2[0]
        fd3 = f4[0]-f3[0]
        fd4 = f5[0]-f4[0]
        fd5 = f6[0]-f5[0]
        fd6 = f7[0]-f6[0]
        fd7 = f8[0]-f7[0]
        fd8 = f9[0]-f8[0]
        fd9 = f10[0]-f9[0]
        fd10 = f11[0]-f10[0]
        fd11 = f12[0]-f11[0]
        fd12 = f13[0]-f12[0]
        fd13 = f14[0]-f13[0]
        fd14 = f15[0]-f14[0]

        return torch.stack(( 
            f0[0], f1[0], f2[0], f3[0], f4[0], f5[0], f6[0], f8[0], f10[0], f12[0], f14[0],
            fd0,   fd1,   fd2,   fd3,   fd4,   fd5,   fd6,   fd8,   fd10,   fd12,   fd14,
            f0[1][0],     f2[1][0],     f4[1][0],     f6[1][0], f8[1][0], f10[1][0],
            f0[1][1],     f2[1][1],     f4[1][1],     f6[1][1], f8[1][1], f10[1][1]
        ))


    def getFrame(self, id):
        if id in self.cache.keys():
            return self.cache[id]

        elif self.preload :
            if id < len(self.data):
                result = [self.data[id],self.time_data[id]]
                self._add_to_cache(id, result)
                return result
            else :
                return self.getFrame(len(self.data)-1)

        else:
            valid, previous = self._get_frame(id-1)
            if valid :

                valid, current = self._get_frame(id)
                if not valid:
                    current = previous
                valid, next_1 = self._get_frame(id+1)
                if not valid :
                    next_1 = current
                valid, next_2 = self._get_frame(id+2)
                if not valid :
                    next_2 = next_1
                valid, next_3 = self._get_frame(id+3)
                if not valid :
                    next_3 = next_2

                window = [
                    self._get_motionFlow(id, previous, current),
                    self._get_motionFlow(id+1, current, next_1),
                    self._get_motionFlow(id+2, next_1, next_2),
                    self._get_motionFlow(id+3, next_2, next_3)
                ]
                res_img = self._motionFlow_to_rgb(previous, window)

                result = [ torch.FloatTensor(current/255)  , torch.FloatTensor(res_img/255) ]

                self._add_to_cache(id, result)

                return result
            else :
                return self.getFrame(self.__len__()-1)


    def getFrameStep(self):
        return self.fps_step

    def _add_to_cache_flow(self, id, value):
        self.cache_flow[id] = value
        if len(self.cache_flow) > self.max_cache_size:
            del self.cache_flow[min(self.cache_flow.keys())]

    def _add_to_cache_frame(self, id, value):
        self.cache_frames[id] = value
        if len(self.cache_frames) > self.max_cache_size:
            del self.cache_frames[min(self.cache_frames.keys())]


    def _add_to_cache(self, id, value):
        self.cache[id] = value
        if len(self.cache) > self.max_cache_size:
            del self.cache[min(self.cache.keys())]


    def _get_frame(self, id):
        if id in self.cache_frames.keys():
            return True, self.cache_frames[id]

        if id != self.previous_id+1 :
            self.video.set(cv2.CAP_PROP_POS_FRAMES, (id-1)*self.fps_step + self.start_frame)
        self.previous_id = id

        for i in range(self.fps_step):
            valid, frame = self.video.read()

        if valid:
            frame = self._process_frame(frame)
            self._add_to_cache_frame(id, frame)
            return valid, frame
        else:
            return valid, None


    def _process_frame(self, frame):
        return cv2.cvtColor( cv2.resize(frame, (self.output_width,self.output_height)) , cv2.COLOR_BGR2GRAY)


    def _get_motionFlow(self, id, previous, next):
        if id in self.cache_flow.keys():
            return self.cache_flow[id]

        previous = cv2.GaussianBlur( previous,(3,3),0)
        next     = cv2.GaussianBlur( next,    (3,3),0)
        result = cv2.calcOpticalFlowFarneback(previous, next, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        self._add_to_cache_flow(id, result)
        return result


    def _motionFlow_to_rgb(self, previous, frames_window):
        flow = (frames_window[0]+frames_window[1]+frames_window[2]+frames_window[3])/4 
        mag, ang = cv2.cartToPolar(np.around(flow[..., 0], decimals=2), np.around(flow[..., 1], decimals=2))
        hsv = np.zeros((2, previous.shape[0], previous.shape[1]), dtype=np.float32)
        hsv[0, ...] = ang*(255/(2*np.pi))
        hsv[1, ...] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return hsv

