# detr-ciou
(Unofficial) Complete IOU loss and Distance IOU loss for DETR transformer object detection


## Instructions for use
all that is needed is a one line change to point it to the new loss function:

1 - Open detr/models/detr.py

2 - Look for the function loss_boxes, (approx line 159 as of this writing)

3 - comment out this line:          loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(

4 - insert this line in its place:  loss_giou = 1 - torch.diag(box_ops.complete_box_iou(

5 - make sure it looks syntactically correct, then save
