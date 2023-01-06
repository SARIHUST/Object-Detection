import torch

from iou import intersection_over_union

def non_maximum_suppression(bboxes, iou_threshold, prob_threshold, bbox_format='midpoint'):
    '''
    Parameters:
        bboxes (list) [[class_label, probability, x1, y1, x2, y2(or x, y, w, h)], ...]: list of Bounding boxes
        iou_threshold: hyper-parameter used to suppress the Bounding boxes of the same target
        prob_threshold: used to select Bounding boxes with high enough probability
        bbox_format: used to determine the IoU computing procedure

    Returns:
        list: list of Bounding boxes after NMS
    '''
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_nms = []

    while bboxes:
        select_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != select_box[0]
            or intersection_over_union(
                torch.tensor(select_box[2:]), torch.tensor(box[2:]), bbox_format=bbox_format
            ) < iou_threshold
        ]

        bboxes_nms.append(select_box)

    return bboxes_nms