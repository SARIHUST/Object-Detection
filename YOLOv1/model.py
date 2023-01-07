import torch
import torch.nn as nn

# the YOLO architecture's convolutional layers according to the paper
YOLO_conv_architecture = [
    (7, 64, 2, 3),  # tuple stands for a convolution layer (kernel_size, out_channels, stride, padding)
    'Maxpool',
    (3, 192, 1, 1),
    'Maxpool',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'Maxpool',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],    # list of tuples(layer groups) and the time of repetition
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'Maxpool',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.norm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels, split_size=7, num_boxes=2, num_classes=20) -> None:
        super().__init__()
        self.conv_architecture = YOLO_conv_architecture
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fc_layers()

    def _create_conv_layers(self):
        conv_layers = []
        in_channels = self.in_channels  # initial in_channels

        for x in self.conv_architecture:
            if type(x) == tuple:    # single conv layer (kernel_size, out_channels, stride, padding)
                conv_layers.append(ConvBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]))
                in_channels = x[1]  # change the in_channels to out_channels for the next layer

            elif type(x) == str:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif type(x) == list:
                conv1, conv2, repeat = x
                for _ in range(repeat):
                    conv_layers.append(ConvBlock(in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    in_channels = conv1[1]
                    conv_layers.append(ConvBlock(in_channels, out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]

        return nn.Sequential(*conv_layers)


    def _create_fc_layers(self):
        S, B, C = self.split_size, self.num_boxes, self.num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),  # this implementation is for the case when the input image has the size of 448
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))  # the 0th dimension is for the batch size

if __name__ == '__main__':
    model = Yolov1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)
    batch_img = torch.randn(5, 3, 448, 448)
    output =  model(batch_img)
    print(output.shape)