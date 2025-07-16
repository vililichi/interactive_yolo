import numpy as np
import torch
import cv2
import math

class Question():
    def __init__(self, mask:np.ndarray, embedding:torch.tensor, mask_conf:float, explain_score:float, image_shape:tuple):
        self.mask = mask.astype(np.bool_)
        self.embedding = embedding
        self.mask_conf = mask_conf
        self.explain_score = explain_score
        self.image_shape = image_shape  

    def get_bbox(self)->tuple:
        xs = np.any(self.mask, axis=1)
        ys = np.any(self.mask, axis=0)
        y1, y2 = np.where(ys)[0][[0, -1]]
        x1, x2 = np.where(xs)[0][[0, -1]]

        return (x1, x2, y1, y2)
    
    def get_centering_score(self)->float:

        img_center_y = self.image_shape[0]/2
        img_center_x = self.image_shape[1]/2

        (x1, x2, y1, y2) =  self.get_bbox()
        center_y = (y1+y2)/2
        center_x = (x1+x2)/2

        dist = math.sqrt((center_x-img_center_x)**2 + (center_y-img_center_y)**2)
        norm_value = math.sqrt(img_center_y**2 + img_center_x**2)

        return 1.0 - max((dist / (norm_value+0.00001)),0)

    def create_image(self, ref_image:np.ndarray)->np.ndarray:

        cv_image = ref_image.copy()
        
        img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img_bgr_mask = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_gray = img_gray.copy()
        img_gray[:] = [0]
        img_gray[self.mask] = [255]

        _, thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        color_a = [0,0,0]
        color_b = [0,0,255]

        cv2.drawContours(img_bgr_mask, contours, -1, color_a, 11)
        cv2.drawContours(img_bgr_mask, contours, -1, color_b, 7)
        cv2.drawContours(img_bgr_mask, contours, -1, color_a, 3)
        img_bgr_mask[self.mask] = cv_image[self.mask]

        (x1, x2, y1, y2) =  self.get_bbox()

        y1 = max(y1-100, 0)
        y2 = min(y2+100, img_bgr_mask.shape[1]-1)
        x1 = max(x1-100, 0)
        x2 = min(x2+100, img_bgr_mask.shape[0]-1)

        img_bgr_mask = img_bgr_mask[x1:x2, y1:y2, :]

        return img_bgr_mask