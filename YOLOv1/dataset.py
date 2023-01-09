import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, labels_dir, S=7, B=2, C=20, transforms=None) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.df.iloc[index, 0])
        img = Image.open(img_path)

        label_path = os.path.join(self.labels_dir, self.df.iloc[index, 1])
        targets = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [
                    float(x) for x in label.replace('\n', '').split()
                ]
                targets.append([class_label, x, y, w, h])

        if self.transforms:
            targets = torch.tensor(targets)
            img, targets = self.transforms(img, targets)

        # the above coordinates are relative to the whole image
        # to run Yolo we need to convert it to the relative of cells
        label_mat = torch.zeros((self.S, self.S, self.C + self.B * 5))  # only the first 5 of the last dimension are useful
        for target in targets:
            class_label, x, y, w, h = target.tolist()
            class_label = int(class_label)

            i, j = int(self.S * x), int(self.S * y)   # determine which cell (x, y) belongs to            
            x_cell, y_cell = self.S * x - i, self.S * y - j # compute the relative distance to the cell

            w_cell, h_cell = w * self.S, h * self.S

            # NOTE: Yolov1 only assigns one object to each cell
            if label_mat[i, j, 20] == 0:
                label[i, j, 20] = 1
                box_coord = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_mat[i, j, 21:25] = box_coord
                label_mat[i, j, class_label] = 1    # similar to one-hot encoding

        return img, label_mat

