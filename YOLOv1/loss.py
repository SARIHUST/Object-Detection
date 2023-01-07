import torch
import torch.nn as nn
from utils import intersection_over_union

class criterion(nn.Module):
    def __init__(self, S=7, B=2, C=20) -> None:
        '''
        Parameters:
            S is the number of grids in each row and column
            B is the number of bounding box each grid predicts
            C is the number of classes (20 for the VOC dataset)
        '''
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')  # reduce it to the sum of the result tensor
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        '''
        Parameters:
            predictions (tensor) [N, S * S * (C + B * 5)]: the result of the network, should be reshaped to [N, S, S, C + B * 5]
            The last dimension of reshaped predictions is [class_scores(0-19), box1(20-24), box2(25-29)] for default
            targets (tensor) [N, S, S, C + 5]: the ground truth of each grid
            The last dimension of targets is [class_gt(0-19), box(20-24)]
        '''
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25], bbox_format='midpoint')
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25], bbox_format='midpoint')
        iou_cat = torch.cat((iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)))
        _, best_box = torch.max(iou_cat, dim=0) 
        # best_box (tensor): [N, S, S, 1]
        # last dimension is 0 -> box1 is better otherwise box2 is better

        exists_target = targets[..., 20].unsqueeze(3)
        # [N, S, S, 1] determines if a grid has a object target in it (the center point of that object is in the grid)

        # NOTE Coordinate Losses (center loss and size loss together)