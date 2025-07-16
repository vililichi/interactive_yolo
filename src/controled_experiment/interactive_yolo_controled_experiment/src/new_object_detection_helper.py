import numpy as np
from ultralytics.utils.ops import scale_image
import torch
from typing import List, Tuple

class NewObjectDetectionParameters():
    def __init__(self):
        self.model_yolo_result = None
        self.model_yolo_result_score_exponant = None
        self.model_sam_result = None
        self.explain_iou_exp = 1.0
        self.explain_conf_exp = 1.2
        self.min_mask_size = 0
        self.min_mask_relative_mask_size = 0.0
        self.min_mask_score = 0.25
        self.max_explication_score = 0.6
    
def new_object_detection(parameters:NewObjectDetectionParameters):
    
    # Run inference
    results_sam = parameters.model_sam_result
    results_yolo = parameters.model_yolo_result

    # Extract result
    masks_sam = [ x for x in torch.unbind(extract_resized_masks(results_sam))]
    confs_sam = [ x.item() for x in torch.unbind(results_sam.boxes.conf)]
    labels_sam = [ "[ CONF = {conf:.2f} ]".format(conf = x) for x in confs_sam]

    w, h = masks_sam[0].shape
    mask_size_threshold = max(parameters.min_mask_size, int(parameters.min_mask_relative_mask_size * w * h ))
    valid_masks_sam = []
    valid_confs_sam = []
    valid_labels_sam = []
    for i in range(len(masks_sam)):
        if masks_sam[i].count_nonzero() > mask_size_threshold and confs_sam[i] > parameters.min_mask_score:
            valid_masks_sam.append(masks_sam[i])
            valid_confs_sam.append(confs_sam[i])
            valid_labels_sam.append(labels_sam[i])
    masks_sam = valid_masks_sam
    confs_sam = valid_confs_sam
    labels_sam = valid_labels_sam

    masks_yolo = []
    confs_yolo = []
    if( results_yolo.masks is not None):
        masks_yolo = [ x for x in torch.unbind(extract_resized_masks(results_yolo))]
        if parameters.model_yolo_result_score_exponant is not None:
            confs_yolo = [ x.item() ** parameters.model_yolo_result_score_exponant[int(results_yolo.boxes.cls[i].item())] for i, x in enumerate(torch.unbind(results_yolo.boxes.conf))]
        else:
            confs_yolo = [ x.item() for x in torch.unbind(results_yolo.boxes.conf)]

    # Try explain
    explication_id, explication_score = try_explain(masks_sam, masks_yolo, confs_yolo, iou_exp=parameters.explain_iou_exp, conf_exp=parameters.explain_conf_exp)

    unexplained_masks  = []
    unexplained_confs  = []
    unexplained_estimation_category_label = []
    unexplained_estimation_conf = []
    unexplained_labels = []
    for i in range(len(masks_sam)):
        if explication_id[i] == -1:
            label = " [ UNEXPLAINED ]".format(conf = confs_sam[i]) 
            unexplained_masks.append(masks_sam[i])
            unexplained_confs.append(confs_sam[i])
            unexplained_labels.append(label)
            unexplained_estimation_category_label.append(None)
            unexplained_estimation_conf.append(0.0)
        elif explication_score[i] < parameters.max_explication_score:
            label = " [ LABEL = {label}, LABEL_CONF = {label_conf:.2} ]".format(label=results_yolo.names[results_yolo.boxes.cls[explication_id[i]].item()], label_conf=explication_score[i]) 
            unexplained_masks.append(masks_sam[i])
            unexplained_confs.append(confs_sam[i])
            unexplained_labels.append(label)
            unexplained_estimation_category_label.append(results_yolo.names[results_yolo.boxes.cls[explication_id[i]].item()])
            unexplained_estimation_conf.append(explication_score[i])

    return (unexplained_masks, unexplained_confs, unexplained_estimation_category_label, unexplained_estimation_conf)

def try_explain(masks_to_explain:List[torch.Tensor], explicative_masks:List[torch.Tensor], explicative_confs:List[float], iou_exp:float = 1.0, conf_exp:float = 0.5)->Tuple[List[int], List[float]]:

    out_id = []
    out_score = []

    for mask in masks_to_explain:
        
        best_id = -1
        best_score = 0.0

        for id in range(len(explicative_masks)):
                explication = explicative_masks[id]
                intersection = torch.count_nonzero(torch.logical_and(mask, explication))
                union = torch.count_nonzero(torch.logical_or(mask, explication)).item()
                iou = float(intersection) / float(union)
                score = (iou ** iou_exp) * (explicative_confs[id] ** conf_exp)
                if score > best_score:
                    best_score = score
                    best_id = id
        
        out_id.append(best_id)
        out_score.append(best_score)

    return out_id, out_score

def extract_resized_masks(result):
    
    if result.masks is None:
        return None
    
    device = result.masks.data.device
    masks = result.masks.data.cpu().detach().numpy()
    masks = np.moveaxis(masks, 0, -1)
    masks = scale_image(masks, result.masks.orig_shape)
    masks = np.moveaxis(masks, -1, 0)
    masks = torch.from_numpy(masks).to(device=device, dtype=torch.bool)

    return masks
