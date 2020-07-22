# detr-ciou
(Unofficial) Complete IOU loss and Distance IOU loss for DETR transformer object detection




To use this loss function:




In file detr/models/detr.py




Within function loss_boxes()

(approximately line 159)




#



# Instructions for use - a one line change to point it to the loss function is all that is needed:
# 1 - Open detr/models/detr.py
# 2 - Look for the function loss_boxes, (approx line 160 as of this writing)
# 3 - comment out this line: 
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
# 4 - insert this line to replace: 
    {@code loss_giou = 1 - torch.diag(box_ops.complete_box_iou(}
# 5 - make sure it looks syntactically correct, save and enjoy better training
