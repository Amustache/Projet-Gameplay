import copy
import time


from torch import nn
import cv2
import numpy as np
import torch
import torchvision


class NeuralNetwork(nn.Module):
    def __init__(self, device, dimensions, batch_size):
        super(NeuralNetwork, self).__init__()

        self.LEARNING_RATE = 0.0005
        self.WEIGHT_DECAY = 0.00001
        self.DROPOUT = 0.5

        self.input_width, self.input_height = dimensions
        self.device = device
        self.loss_fn = nn.BCELoss()

        self.all_correct = torch.Tensor([True, True, True]).to(device)

        conv_output_width = 5
        conv_output_height = 2

        self.block_init = nn.Sequential(
            nn.Conv2d(34, 70, 7, padding=3),
            nn.MaxPool2d(4),
        )
        self.block_init.apply(self.init_weights)

        self.block_1 = nn.Sequential(
            nn.Conv2d(70, 35, 3, padding=1),
            nn.BatchNorm2d(35),
            nn.LeakyReLU(),
            nn.Conv2d(35, 35, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
            nn.Conv2d(35, 70, 3, padding=1),
            nn.BatchNorm2d(70),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
        )
        self.block_1.apply(self.init_weights)

        self.block_2 = nn.Sequential(
            nn.Conv2d(70, 35, 3, padding=1),
            nn.BatchNorm2d(35),
            nn.LeakyReLU(),
            nn.Conv2d(35, 35, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
            nn.Conv2d(35, 70, 3, padding=1),
            nn.BatchNorm2d(70),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
        )
        self.block_2.apply(self.init_weights)

        self.block_3 = nn.Sequential(
            nn.Conv2d(70, 35, 3, padding=1),
            nn.BatchNorm2d(35),
            nn.LeakyReLU(),
            nn.Conv2d(35, 35, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
            nn.Conv2d(35, 70, 3, padding=1),
            nn.BatchNorm2d(70),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
        )
        self.block_3.apply(self.init_weights)

        self.block_inter = nn.Sequential(
            nn.Conv2d(70, 100, 3, padding=1),
            nn.Dropout2d(self.DROPOUT),
            nn.MaxPool2d(4),
        )
        self.block_inter.apply(self.init_weights)

        self.block_4 = nn.Sequential(
            nn.Conv2d(100, 50, 3, padding=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
            nn.Conv2d(50, 50, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
            nn.Conv2d(50, 100, 3, padding=1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
        )
        self.block_4.apply(self.init_weights)

        self.block_5 = nn.Sequential(
            nn.Conv2d(100, 50, 3, padding=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
            nn.Conv2d(50, 50, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
            nn.Conv2d(50, 100, 3, padding=1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
        )
        self.block_5.apply(self.init_weights)

        self.block_end = nn.Sequential(
            nn.Conv2d(100, 130, 3, padding=1),
            nn.Dropout2d(self.DROPOUT),
            nn.MaxPool2d(3),
            nn.Conv2d(130, 180, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT),
            nn.Flatten(),
            nn.Linear(180 * conv_output_width * conv_output_height, 180),
            nn.LeakyReLU(),
            nn.Dropout2d(self.DROPOUT / 1.5),
            nn.Linear(180, 3),
            nn.Sigmoid(),
        )
        self.block_end.apply(self.init_weights)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY
        )

        self.to(device)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # Print the shape at each layer
        """
        x   = self.block_init(x)
        x_1 = self.block_1(x)+x
        x_2 = self.block_2(x_1)+x_1
        x_3 = self.block_3(x_2)+x_2

        x_4 = self.block_inter(x_3)
        x_5 = self.block_4(x_4)+x_4
        x_6 = self.block_5(x_5)+x_5
        for layer in self.block_end:
            print(x_6.shape)
            x_6 = layer(x_6 )
        return x_6
        """

        x = self.block_init(x)
        x_1 = self.block_1(x) + x
        x_2 = self.block_2(x_1) + x_1
        x_3 = self.block_3(x_2) + x_2

        x_4 = self.block_inter(x_3)
        x_5 = self.block_4(x_4) + x_4
        x_6 = self.block_5(x_5) + x_5

        return self.block_end(x_6)

    def process(self, data, is_train):
        loss_sum = 0
        fully_correct = 0
        part_correct = torch.FloatTensor([0, 0, 0]).to(self.device)

        nb_elem_per_x = 1 if is_train else 1
        data_size = len(data) * nb_elem_per_x

        self.train() if is_train else self.eval()
        with torch.set_grad_enabled(is_train):
            for batch, (x, y) in enumerate(data):
                x = x.to(self.device)

                # Forward
                pred = self(x)
                loss = self.loss_fn(pred, y)
                loss_sum += loss.item()

                #  Backward
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                corrects = torch.round(pred) == y
                part_correct[0] += corrects[:, 0].sum()
                part_correct[1] += corrects[:, 1].sum()
                part_correct[2] += corrects[:, 2].sum()
                fully_correct += torch.logical_and(
                    torch.logical_and(corrects[:, 0], corrects[:, 1]), corrects[:, 2]
                ).sum()

                if is_train and batch % 50 == 0:
                    print(
                        f"loss:{loss.item():>3f} [{batch:>4d}/{data_size:>4d}] {100*batch/data_size:.1f}%"
                    )

        return (
            loss_sum / data_size,
            part_correct / (len(data.dataset) * nb_elem_per_x),
            fully_correct / (len(data.dataset) * nb_elem_per_x),
        )

    def debug_frame(self, x, y=None, name=""):
        print("DEBUG | X shape : ", x.shape)
        for i in range(5):
            for j in range(5):
                # if y != None : print(y[i])
                cv2.imwrite(
                    "output/image_" + name + str(i) + str(j) + ".png",
                    x[i][j].cpu().detach().numpy() * 255,
                )
