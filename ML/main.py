import sys
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from inc.NeuralNetwork import NeuralNetwork
from inc.VideoLoader import VideoLoader
from inc.KeyslogReader import KeyslogReader

from torch.utils.tensorboard import SummaryWriter

DEVICE           = "cuda:0" if torch.cuda.is_available() else "cpu"
VIDEO_DIMENSIONS = (int(1920/8), int(1080/8))
START_FRAME      = 1000
END_FRAME        = 0
OFFSET           = 3
BATCH_SIZE       = 40
NB_EPOCHS        = 100
NB_K_SPLIT       = 8
SHUFFLE_DATASETS = False
FPS              = 25


class VideoKeysLogMerge(torch.utils.data.IterableDataset):
    def __init__(self, VideoLoader, KeyslogReader):
        self.video = VideoLoader
        self.keys = KeyslogReader
        self.next_element = 0
    def __len__(self):        return len(self.video)
    def __iter__(self):       return self
    def __getitem__(self, i): return self.video[i], self.keys[i]
    def __next__(self):
        if self.next_element < self.__len__():
            self.next_element += 1
            return self.__getitem__(self.next_element-1)
        else:
            self.next_element = 0
            raise StopIteration

def start_tensorboard():
    global tensorboardWriter
    tensorboardWriter = SummaryWriter(comment="Neural Network")

def write_tensorboard(name, epoch, results):
    global tensorboardWriter
    loss, correct_parts, correct = results
    tensorboardWriter.add_scalar(name+"/Accuracy/buttonLeft",  correct_parts[0], epoch)
    tensorboardWriter.add_scalar(name+"/Accuracy/buttonRight", correct_parts[1], epoch)
    tensorboardWriter.add_scalar(name+"/Accuracy/buttonJump",  correct_parts[2], epoch)
    tensorboardWriter.add_scalar(name+"/Accuracy",             correct, epoch)
    tensorboardWriter.add_scalar(name+"/Loss",                 loss, epoch)
    tensorboardWriter.flush()

def save_model(model, suffix):
    torch.save(model.state_dict(), datetime.now().strftime("%Y-%m-%d-%H:%M")+suffix)

def get_video(path, preload):
    return VideoLoader(path, VIDEO_DIMENSIONS, START_FRAME, END_FRAME, FPS, preload)

def get_keylog(path, frameStep):
    return KeyslogReader(path, frameStep, START_FRAME, offset=OFFSET)

def get_model():
    return NeuralNetwork(DEVICE, VIDEO_DIMENSIONS)

def get_dataloader(data):
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASETS)

def print_epoch(epoch):
    print(f"--------------  Epoch {epoch+1}  -----------------")

def test(video_path, keylog_path):
    print("Start testing...")
    start_tensorboard()
    data_frames = get_video(video_path, preload=True)
    data_keys   = get_keylog(keylog_path, data_frames.getFrameStep())
    data        = VideoKeysLogMerge(data_frames, data_keys)

    kf = KFold(n_splits=NB_K_SPLIT).split(data)
    for split, (train_indexes, test_indexes) in enumerate(kf):

        data_train = get_dataloader(Subset(data, train_indexes))
        data_test  = get_dataloader(Subset(data, test_indexes))
        model      = get_model()

        for epoch in range(NB_EPOCHS):
            print_epoch(epoch)
            results = model.process(data_train, is_train=True)
            write_tensorboard("Train", epoch+split*NB_EPOCHS, results)
            results = model.process(data_test, is_train=False)
            loss, correct_parts, correct = results
            print(f"Test : Accuracy:{(100*correct_parts[0]):>0.1f}%|{(100*correct_parts[1]):>0.1f}%|{(100*correct_parts[2]):>0.1f}% Tot: {(100*correct):>0.1f}% Loss: {loss:>6f} \n")
            write_tensorboard("Test", epoch+split*NB_EPOCHS, results)

        save_model(model, "_test_model_tmp.pth")

    save_model(model, "_test_model_finished.pth")
    print("Test finished")


def train(video_path, keylog_path):
    print("Start training...")
    start_tensorboard()
    data_frames = get_video(video_path, preload=False)
    data_keys   = get_keylog(keylog_path, data_frames.getFrameStep())
    data        = get_dataloader(VideoKeysLogMerge(data_frames, data_keys))
    model       = get_model()
    for epoch in range(NB_EPOCHS):
        print_epoch(epoch)
        write_tensorboard("Train", epoch*NB_EPOCHS, model.process(data, is_train=True))
    save_model(model, "_train_model.pth")
    print("Train finished")


def predict(video_path, model_path):
    if os.path.isfile(video_path):
        print("Predicting "+video_path)
        keys_dict = [
            "Key.left",
            "Key.right",
            "x"
        ]
        current_state = [0,0,0]

        model = get_model()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        data = get_video(video_path, preload=False)
        dataloader = get_dataloader(data)

        frame = START_FRAME
        f = open( (os.path.basename(video_path).split(".")[0]) +".csv", "w")
        f.write("FRAME,KEY,STATUS\n")
        with torch.no_grad():
            i = 0
            for x in dataloader:
                print(f"{i}/{len(dataloader)}")
                x[0] = x[0].to(DEVICE)
                x[1] = x[1].to(DEVICE)
                pred_batch = torch.round(model(x))
                for pred in pred_batch:
                    for j in range(len(current_state)):
                        if current_state[j] != pred[j]:
                            state = ("DOWN" if pred[j]==1 else "UP")
                            f.write(f"{frame},{keys_dict[j]},{state}\n")
                            current_state[j] = pred[j]
                    frame += data.fps_step
                i+=1
        f.close()
        print("Prediction finished")

    elif os.path.isdir(video_path):
        files = os.listdir(video_path)
        for f in files:
            predict(video_path+f, model_path)
    else:
        print("Path is not a file or a directory")


def main():
    if len(sys.argv) < 2: print("No action specified !")
    elif sys.argv[1] == "train"   : train(sys.argv[2],   sys.argv[3])
    elif sys.argv[1] == "test"    : test(sys.argv[2],    sys.argv[3])
    elif sys.argv[1] == "predict" : predict(sys.argv[2], sys.argv[3])
    else: print("Unknown command")

if __name__ == "__main__": main()