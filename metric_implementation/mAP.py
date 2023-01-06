import torch
from iou import intersection_over_union

def mean_average_percision(pred_boxes, gt_boxes, iou_threshold=0.5, bbox_format='midpoint', num_classes=20):
    '''
    Calculates the mean average precision at a certain iou_threshold

    Parameters:
        pred_boxes (list): list of predicted bounding boxes, [[img_idx, pred_class, prob_score, x, y, w, h], ...]
        gt_boxes (list): same with pred_boxes, but the values are the ground truth
        iou_threshold: threshold where predicted bounding boxes are correct
        box_format: midpoint or corners used in iou computation
        num_classes: number of target classes

    Returns:
        mAP values across all classes under a certain IoU threshold
    '''
    average_percisions = []

    predictions, ground_truths = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    for c in range(num_classes):
        for pbox in pred_boxes:
            if pbox[1] == c:
                predictions[c].append(pbox)
        for tbox in gt_boxes:
            if tbox[1] == c:
                ground_truths[c].append((tbox, len(ground_truths[c])))
                # add the index of the ground truth box, later use it to determine if a ground truth box is matrched

    for c in range(num_classes):
        total_gt_num = len(ground_truths[c])
        if total_gt_num == 0:
            continue    # don't count this class if all the images in the dataset doesn't contain this type of targets
        
        gt_matched = torch.zeros(len(ground_truths[c])) # determine if the ground truth targets are matched
        TP = torch.zeros(len(predictions[c]))
        FP = torch.ones(len(predictions[c]))
        predictions[c].sort(key=lambda x: x[2], reverse=True)   # sort the bounding boxes according to the probability score
        
        for pred_idx, pred_box in enumerate(predictions[c]):
            img_idx = pred_box[0]
            img_gt_boxes = [gt_box for gt_box in ground_truths[c] if gt_box[0][0] == img_idx]

            best_iou, best_idx = 0, 0

            for (gt_box, gt_idx) in img_gt_boxes:
                if gt_matched[gt_idx] == 1:
                    continue    
                    # this target has already been matched to another bounding box with a higher probability score
                    # so this pred_box is a FP, no need to change TP[pred_idx] or FP[pred_idx]
                iou = intersection_over_union(
                    torch.tensor(pred_box[3:]),
                    torch.tensor(gt_box[3:]),
                    bbox_format=bbox_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx

            if best_iou > iou_threshold:
                # this pred_box is a TP
                gt_matched[best_idx] = 1
                TP[pred_idx] = 1
                FP[pred_idx] = 0

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        precision = TP_cumsum / (TP_cumsum + FP_cumsum)
        recall = TP_cumsum / total_gt_num
        precision = torch.cat((torch.tensor([1]), precision))   # the graph of P-R starts from (0, 1), add 1 to precsion
        recall = torch.cat((torch.tensor([0]), recall))         # add 0 to recall
        AP = torch.trapezoid(y=precision, x=recall)             # use torch.trapezoid to compute the approximate area under the P-R line
        average_percisions.append(AP)

    return sum(average_percisions) / len(average_percisions)