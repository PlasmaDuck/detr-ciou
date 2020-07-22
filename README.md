# detr-ciou
(Unofficial) Complete IOU loss and Distance IOU loss for DETR transformer object detection


To use this loss function:

In file detr/models/detr.py

Within function loss_boxes()
(approximately line 159)

#loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
loss_giou = 1 - torch.diag(box_ops.complete_box_iou(
