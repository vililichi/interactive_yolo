import numpy as np
import cv2
import random
import math
from ultralytics.utils.ops import scale_image
import torch
from typing import List, Tuple

class NewObjectDetectionParameters():
    def __init__(self):
        self.model_yolo_result = None
        self.model_sam_result = None
        self.explain_iou_exp = 1.0
        self.explain_conf_exp = 1.2
        self.min_mask_size = 0
        self.min_mask_relative_mask_size = 0.0
    
def new_object_detection(cv_img:np.ndarray, parameters:NewObjectDetectionParameters) -> Tuple[Tuple[List[torch.Tensor], List[float], List[str]], Tuple[List[torch.Tensor], List[float], List[str]], Tuple[List[torch.Tensor], List[float], List[str]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
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
        if masks_sam[i].count_nonzero() > mask_size_threshold:
            valid_masks_sam.append(masks_sam[i])
            valid_confs_sam.append(confs_sam[i])
            valid_labels_sam.append(labels_sam[i])
    masks_sam = valid_masks_sam
    confs_sam = valid_confs_sam
    labels_sam = valid_labels_sam

    masks_yolo = []
    confs_yolo = []
    labels_yolo = []
    if( results_yolo.masks is not None):
        masks_yolo = [ x for x in torch.unbind(extract_resized_masks(results_yolo))]
        confs_yolo = [ x.item() for x in torch.unbind(results_yolo.boxes.conf)]
        labels_yolo = [ " [ CONF = {conf:.2f}, NAME = {name} ]".format(conf = confs_yolo[i], name=results_yolo.names[results_yolo.boxes.cls[i].item()]) for i in range(len(confs_yolo)) ]

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
        elif explication_score[i] < 0.6:
            label = " [ LABEL = {label}, LABEL_CONF = {label_conf:.2} ]".format(label=results_yolo.names[results_yolo.boxes.cls[explication_id[i]].item()], label_conf=explication_score[i]) 
            unexplained_masks.append(masks_sam[i])
            unexplained_confs.append(confs_sam[i])
            unexplained_labels.append(label)
            unexplained_estimation_category_label.append(results_yolo.names[results_yolo.boxes.cls[explication_id[i]].item()])
            unexplained_estimation_conf.append(explication_score[i])

    # save imgs
    img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_mask_sam = draw_masks(img_gray_bgr, masks_sam, confs_sam, labels_sam, conf_threshold=0.8)
    img_mask_yolo = draw_masks(img_gray_bgr, masks_yolo, confs_yolo, labels_yolo, conf_threshold=0.57)
    img_mask_unexplained = draw_masks(img_gray_bgr, unexplained_masks, unexplained_confs, unexplained_labels, conf_threshold=0.9)

    return ((unexplained_masks, unexplained_confs, unexplained_estimation_category_label, unexplained_estimation_conf),(img_mask_sam, img_mask_yolo, img_mask_unexplained))

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

def draw_masks(image, masks, confs, labels=None, conf_threshold=0.0):
    """
    Draws masks on the image with random colors.
    """
    img_out = image.copy()

    if len(masks) == 0:
        return img_out
    
    h_0 = img_out.shape[0]
    w_0 = img_out.shape[1]

    nbr_label = 0
    max_label_len = 0
    nbr_label_per_row = 1
    w_size_per_label = 0
    h_size_per_label = 0
    h_added = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontThickness = 3

    h_added = 100
    if labels is not None:

        nbr_label = len(labels)
        max_label_len = max([len(x) for x in labels])

        h_size_per_label = 40
        w_size_per_char = 20
        w_size_per_label = (max_label_len+3) * w_size_per_char
        nbr_label_per_row = math.floor(w_0 / w_size_per_label)

        while nbr_label_per_row == 0:
            fontScale = fontScale * 0.8
            fontThickness = fontThickness * 0.8

            h_size_per_label = h_size_per_label * 0.8
            w_size_per_char = w_size_per_char * 0.8

            w_size_per_label = (max_label_len+3) * int(w_size_per_char)
            nbr_label_per_row = math.floor(w_0 / w_size_per_label)
        
        fontThickness = int(fontThickness)
        h_size_per_label = int(h_size_per_label)
        w_size_per_char = int(w_size_per_char)
            

        h_added = 8 + (math.ceil(nbr_label / nbr_label_per_row) * h_size_per_label)

        new_img_out = np.zeros((h_0+h_added, w_0, 3), dtype=np.uint8)+255
        new_img_out[:h_0, :w_0, :] = img_out
        img_out = new_img_out

    used_id = []
    for i in range(len(masks)):
        if confs[i] > conf_threshold:
            used_id.append(i)

    if(len(used_id) == 0):
        return img_out
    
    color_map_coef = 0.5
    if(len(used_id) > 1):
        color_map_coef = 1.0 / (len(used_id)-1)

    for i in range(len(used_id)):
        mask = masks[i]

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy()
        mask = mask.astype(bool)

        conf = confs[i]

        if labels is not None:
            new_mask = np.zeros((h_0+h_added, w_0), dtype=bool)
            new_mask[:h_0, :w_0] = mask
            mask = new_mask

        

        img_out_alt = img_out.copy()
        color_gray_Value = int(255*i*color_map_coef)
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8([[[color_gray_Value]]]), cv2.COLORMAP_JET),cv2.COLOR_RGB2BGR)[0][0] 
        img_out_alt[mask] = color

        alpha = 0.25
        beta = 1.0 - alpha

        img_out = cv2.addWeighted(img_out, alpha, img_out_alt, beta, 0)

        if labels is not None:
            org_x_o = (i % nbr_label_per_row) * w_size_per_label
            org_x_1 = ((i % nbr_label_per_row) * w_size_per_label) + (3*w_size_per_char)
            org_y = h_0 + h_size_per_label + (h_size_per_label * (i // nbr_label_per_row))
            org_o = (org_x_o, org_y)
            org_1 = (org_x_1, org_y)

            img_out = cv2.putText(img_out, "[@]", org_o, font, 
                   fontScale, color.tolist(), fontThickness, cv2.LINE_AA)
            img_out = cv2.putText(img_out, labels[i], org_1, font, 
                   fontScale, (0,0,0), fontThickness, cv2.LINE_AA)

    return img_out

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
