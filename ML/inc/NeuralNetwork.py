import copy


from torch import nn
import cv2
import numpy as np
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, device, dimensions):
        super(NeuralNetwork, self).__init__()
        self.input_width, self.input_height = dimensions
        self.device = device
        self.loss_fn = nn.BCELoss()

        conv_output_width = 2
        conv_output_height = 4

        # self.init_frame = nn.Sequential(
        #    nn.Conv2d(1, 64, 7),
        #    nn.MaxPool2d(5),
        # )
        # self.init_frame.apply(self.init_weights)

        self.init_time = nn.Sequential(
            nn.Conv2d(11, 256, 7),
            nn.MaxPool2d(5),
        )
        self.init_time.apply(self.init_weights)

        self.forward_1 = nn.Sequential(
            # nn.Conv2d(192, 64, 1),
            nn.MaxPool2d(3),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(3),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(256 * conv_output_width * conv_output_height, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        self.forward_1.apply(self.init_weights)

        """
        self.forward_1 = nn.Sequential(

            nn.Conv2d(192, 32, 1),
            nn.MaxPool2d(8),
            nn.Dropout2d(0.4),

            nn.Flatten(),
            nn.Linear(32*conv_output_width*conv_output_height, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.forward_2 = copy.deepcopy(self.forward_1)
        self.forward_3 = copy.deepcopy(self.forward_1)

        self.forward_1.apply(self.init_weights)
        self.forward_2.apply(self.init_weights)
        self.forward_3.apply(self.init_weights)
        """

        self.optimizer_fn = torch.optim.SGD(self.parameters(), lr=0.016, weight_decay=0.002)
        self.to(device)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):

        frame_rgb, frame_time = x

        # Print the shape at each layer
        # x = x[:, 0:3]
        # for layer in self.init_frame:
        #    print(frame_rgb.shape)
        #    frame_rgb = layer(frame_rgb)
        # for layer in self.init_time:
        #    print(frame_time.shape)
        #    frame_time = layer(frame_time)
        # x = torch.cat((frame_rgb,frame_time), dim=1)
        # for layer in self.forward_1:
        #    print(x.shape)
        #    x = layer(x )
        # return x

        # self.debug_frame(frame_rgb)
        # print(frame_rgb.shape)
        # print(frame_time.shape)
        # self.debug_frame(torch.cat( (
        #    frame_rgb[:, 0],
        #    frame_time[:, 0],
        #    frame_time[:, 1],
        #    frame_time[:, 2],
        #    frame_time[:, 3],
        #    frame_time[:, 4]
        #    ), dim=1 ))

        # init_frame = self.init_frame(frame_rgb)
        init = self.init_time(torch.cat((frame_rgb, frame_time), dim=1))
        # init = torch.cat( (init_frame, init_time), dim=1 )
        # f1 = self.forward_1( init )
        # f2 = self.forward_2( init )
        # f3 = self.forward_3( init )
        # return torch.cat( (f1,f2,f3), dim=1)
        return self.forward_1(init)

    def process(self, data, is_train):
        loss_sum = 0
        fully_correct = 0
        part_correct = torch.FloatTensor([0, 0, 0])

        self.train() if is_train else self.eval()
        with torch.set_grad_enabled(is_train):
            for batch, (x, y) in enumerate(data):
                x[0] = x[0].to(self.device)
                x[1] = x[1].to(self.device)
                y = y.to(self.device)

                # self.debug_frame(x[0],y)

                # Forward
                pred = self(x)
                loss = self.loss_fn(pred, y)
                loss_sum += loss.item()

                if is_train:
                    #  Backward
                    self.optimizer_fn.zero_grad()
                    loss.backward()
                    self.optimizer_fn.step()

                for j in range(len(pred)):
                    for i in range(len(pred[j])):
                        if round(float(pred[j][i])) == int(y[j][i]):
                            part_correct[i] += 1
                    if torch.equal(torch.round(pred[j]), y[j]):
                        fully_correct += 1

                if is_train and batch % 50 == 0:
                    print(
                        f"loss:{loss.item():>3f} [{batch:>4d}/{len(data):>4d}] {100*batch/len(data):.1f}%"
                    )

        return (
            loss_sum / len(data),
            part_correct / len(data.dataset),
            fully_correct / len(data.dataset),
        )

    def debug_frame(self, x, y=None):
        for i in range(x.shape[0]):
            if y != None:
                print(y[i])
            cv2.imshow("image", torch.permute(x[i], (1, 2, 0)).cpu().numpy())
            cv2.waitKey(0)
