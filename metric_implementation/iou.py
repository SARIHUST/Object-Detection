import torch

def intersection_over_union(bbox_pred, bbox_gt, bbox_format='midpoint'):
    '''
    Parameters:
        bbox_pred (tensor): Predictions of the Bounding boxes, shape (batch_size, 4) or (batch_size, x, ..., 4)
        bbox_gt (tensor): Ground Truth of the Bounding boxes, shape (batch_size, 4) or (batch_size, x, ..., 4)
        bbox_format (str): midpoint/corners, if midpoint, then box is (x, y, w, h) else (x1, y1, x2, y2)

    Returns:
        tensor: IoU for all Bounding boxes in the batch
    '''

    # use ... for the possible additional channels
    # use idx:idx + 1 to maintain the tensor dimension
    if bbox_format == 'midpoint':
        pred_x1 = bbox_pred[..., 0:1] - bbox_pred[..., 2:3] / 2
        pred_y1 = bbox_pred[..., 1:2] - bbox_pred[..., 3:4] / 2
        pred_x2 = bbox_pred[..., 0:1] + bbox_pred[..., 2:3] / 2
        pred_y2 = bbox_pred[..., 1:2] + bbox_pred[..., 3:4] / 2
        gt_x1 = bbox_gt[..., 0:1] - bbox_gt[..., 2:3] / 2
        gt_y1 = bbox_gt[..., 1:2] - bbox_gt[..., 3:4] / 2
        gt_x2 = bbox_gt[..., 0:1] + bbox_gt[..., 2:3] / 2
        gt_y2 = bbox_gt[..., 1:2] + bbox_gt[..., 3:4] / 2

    elif bbox_format == 'corners':
        pred_x1 = bbox_pred[..., 0:1]
        pred_y1 = bbox_pred[..., 1:2]
        pred_x2 = bbox_pred[..., 2:3]
        pred_y2 = bbox_pred[..., 3:4]
        gt_x1 = bbox_gt[..., 0:1]
        gt_y1 = bbox_gt[..., 1:2]
        gt_x2 = bbox_gt[..., 2:3]
        gt_y2 = bbox_gt[..., 3:4]

    else:
        print('Unrecognized Bounding box format {}'.format(bbox_format))
        return -1

    x1 = torch.max(pred_x1, gt_x1)
    y1 = torch.max(pred_y1, gt_y1)
    x2 = torch.min(pred_x2, gt_x2)
    y2 = torch.min(pred_y2, gt_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  # use clamp to deal with cases when there is no intersection
    pred_area = abs((pred_x1 - pred_x2) * (pred_y1 - pred_y2))
    gt_area = abs((gt_x1 - gt_x2) * (gt_y1 - gt_y2))

    return intersection / (pred_area + gt_area + 1e-8)