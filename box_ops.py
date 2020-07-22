# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# from @PlasmaDuck: unofficially adds DIoU and CIoU
# courtesy of AudereNow.org

# CIoU proposed in https://arxiv.org/abs/2005.03572v2, 
# "Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation"
# credit Zhaohui Zheng, Ping Wang, Dongwei Ren, Wei Liu, Rongguang Ye, Qinghua Hu, Wangmeng Zuo

# DIoU proposed in https://arxiv.org/abs/1911.08287v1, 
# "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
# credit Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren

# Instructions for use - all that is needed is a one line change to point it to the new loss function:
# 1 - save this file as detr/util/box_ops.py
# 2 - open detr/models/detr.py
# 3 - look for the function loss_boxes, (approx line 159 as of this writing)
# 4 - comment out this line:          loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
# 5 - insert this line in its place:  loss_giou = 1 - torch.diag(box_ops.complete_box_iou(
# 6 - make sure it looks syntactically correct, then save

"""
Utilities for bounding box manipulation and CIoU, DIoU, and GIoU.
"""
import torch
import numpy as np
import math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]


    return iou - (area - union) / area


def distance_box_iou(boxes1, boxes2):
    if boxes1.nelement() == 0:
        return boxes1
    giou = generalized_box_iou(boxes1, boxes2)
    center1 = torch.stack((boxes1[:, 0] + boxes1[:, 2], boxes1[:, 1] + boxes1[:, 3])).transpose(1, 0) / 2
    center2 = torch.stack((boxes2[:, 0] + boxes2[:, 2], boxes2[:, 1] + boxes2[:, 3])).transpose(1, 0) / 2
    center_distance = center1 - center2
    center_distance2 = center_distance ** 2
    center_distance2 = center_distance2.sum(dim=1)

    min_x = torch.min(boxes1[:, ::2], boxes2[:, ::2]).min(dim=1)[0]
    min_y = torch.min(boxes1[:, 1::2], boxes2[:, 1::2]).min(dim=1)[0]
    max_x = torch.max(boxes1[:, ::2], boxes2[:, ::2]).max(dim=1)[0]
    max_y = torch.max(boxes1[:, 1::2], boxes2[:, 1::2]).max(dim=1)[0]
    
    corner_distance2 = ((max_x - min_x) ** 2) + ((max_y - min_y) ** 2)

    return center_distance2 / corner_distance2 + giou


def complete_box_iou(boxes1, boxes2):
    if boxes1.nelement() == 0:
        return boxes1
    diou = distance_box_iou(boxes1, boxes2)
    iou = box_iou(boxes1, boxes2)[0]

    angle1 = torch.atan((boxes1[:, 3]- boxes1[:, 1]) / (boxes1[:, 2] - boxes1[:, 0]))
    angle2 = torch.atan((boxes2[:, 3]- boxes2[:, 1]) / (boxes2[:, 2] - boxes2[:, 0]))
    angle = angle2 - angle1

    value = angle * 4 / (math.pi ** 2)

    factor = value / (1 - iou + value)
    
    return diou + factor * value


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
