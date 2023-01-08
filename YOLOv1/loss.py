'''
This implementation of the YOLO loss is only for the case when each grid outputs 2 bounding boxes
'''

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
        self.B = B  # actually this should be set to 2
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        '''
        This implementation is only for B = 2, consider modifying it later to satisfy other settings
        Parameters:
            predictions (tensor) [N, S * S * (C + 2 * 5)]: the result of the network, should be reshaped to [N, S, S, C + 2 * 5]
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

        '''
        Coordinate Losses (center loss + size loss)
        '''
        select_boxes = exists_target * (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[21:25])   # [N, S, S, 4]
        # if the gird does not contain any target (exists_target for this grid is 0), then set the select_box to all 0s
        # use best_box and 1 - best_box to determine which predicted bounding box is used
        target_boxes = exists_target * targets[..., 21:25]  # [N, S, S, 4]

        # set the (w, h) terms to (sqrt(w), sqrt(h))
        select_boxes[..., 2:4] = torch.sign(select_boxes[..., 2:4]) * torch.sqrt(torch.abs(select_boxes[..., 2:4]) + 1e-6)
        # use sign and abs because in the first few epochs, the predictions might output negative w and h
        target_boxes[..., 2:4] = torch.sqrt(target_boxes[..., 2:4])
        
        coord_loss = self.mse(
            torch.flatten(select_boxes, end_dim=-2),    # [N * S * S, 4] 
            torch.flatten(target_boxes, end_dim=-2)     # end_dim=-2 means the flatten process ends at the second to last dimension
        )

        '''
        Confidence Losses (object loss + no object loss)
        '''
        # object loss only use the best bounding box
        select_confidence = best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]   # [N, S, S, 1]
        target_confidence = targets[..., 20:21]

        object_loss = self.mse(
            torch.flatten(exists_target * select_confidence),   # [N * S * S * 1]
            torch.flatten(exists_target * target_confidence)
        )

        # no object loss use both the bounding boxes
        no_object_loss = self.mse(
            torch.flatten((1 - exists_target) * predictions[..., 20:21]),
            torch.flatten((1 - exists_target) * targets[..., 20:21])
        ) + self.mse(
            torch.flatten((1 - exists_target) * predictions[..., 25:26]),
            torch.flatten((1 - exists_target) * targets[..., 20:21])
        )

        '''
        Class Loss
        '''
        class_loss = self.mse(
            torch.flatten(exists_target * predictions[..., :20], end_dim=-2),   # [N * S * S, 20]
            torch.flatten(exists_target * targets[..., :20], end_dim=-2)
        )

        '''
        Final Loss
        '''

        loss = (
            self.lambda_coord * coord_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss