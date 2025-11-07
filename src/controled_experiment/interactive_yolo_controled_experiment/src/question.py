import numpy as np
import torch
import cv2
import math

class Question():
    def __init__(self, mask:np.ndarray, embedding:torch.tensor, mask_conf:float, explain_score:float, image_shape:tuple, bbox:tuple = None):
        self.mask = mask.astype(np.bool_)
        self.embedding = embedding
        self.mask_conf = mask_conf
        self.explain_score = explain_score
        self.image_shape = image_shape
        self.bbox = bbox
        self.relative_size = None

    def get_bbox(self)->tuple:

        if self.bbox is not None:
            return self.bbox
        
        ys = np.any(self.mask, axis=1)
        xs = np.any(self.mask, axis=0)
        y1, y2 = np.where(ys)[0][[0, -1]]
        x1, x2 = np.where(xs)[0][[0, -1]]
        self.bbox = (x1, y1, x2, y2)

        return self.bbox
    
    def get_centering_score(self)->float:

        img_center_y = self.image_shape[0]/2
        img_center_x = self.image_shape[1]/2

        (x1, y1, x2, y2) =  self.get_bbox()
        center_y = (y1+y2)/2
        center_x = (x1+x2)/2

        dist = math.sqrt((center_x-img_center_x)**2 + (center_y-img_center_y)**2)
        norm_value = math.sqrt(img_center_y**2 + img_center_x**2)

        return max(1.0 - (dist / (norm_value+0.00001)),0)
    
    def get_relative_size(self):

        if self.relative_size is None:
            size = np.count_nonzero(self.mask)
            img_size = self.image_shape[0]*self.image_shape[1]
            self.relative_size = float(size)/float(img_size)

        return self.relative_size
    
    def get_size_score(self, min_optimal_size = 0.1, max_optimal_size = 0.5)->float:

        relative_size = self.get_relative_size()
        if relative_size >= min_optimal_size and relative_size <= max_optimal_size:
            return 1.0

        diff_optimal_size = 1.0 - min(abs(min_optimal_size - relative_size), abs(max_optimal_size - relative_size))
        return diff_optimal_size


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

        (x1, y1, x2, y2) =  self.get_bbox()

        y1 = max(y1-100, 0)
        y2 = min(y2+100, img_bgr_mask.shape[0]-1)
        x1 = max(x1-100, 0)
        x2 = min(x2+100, img_bgr_mask.shape[1]-1)

        img_bgr_mask = img_bgr_mask[y1:y2, x1:x2, :]

        return img_bgr_mask