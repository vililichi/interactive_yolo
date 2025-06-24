import os

import pycocotools.coco
from torchmetrics.functional.detection.map import mean_average_precision

import rclpy
from rclpy.node import Node
from database_service import DatabaseServices
from ...utils.embedding_generator import EmbeddingGenerator
from ...utils.model_loader.base_model import base_model
from ...utils.new_object_detection_helper import extract_resized_masks
from interactive_yolo_utils import workspace_dir
from threading import Thread

import torch
import numpy as np
import json
import time

import cv2
from matplotlib import pyplot as plt

import pycocotools
import numpy as np

from typing import Tuple, List

from .rle_decode import rle_to_mask

class TaskNode(Node):
    def __init__(self):
        super().__init__('embedding_distance_histogram_node')

        self.database_services = DatabaseServices(self)
        self.embeddings_generator = EmbeddingGenerator(self.database_services)
        self.ref_model = base_model()
        self.trained_model = base_model()
        self.trained_model_with_no_classes_specification = base_model()

        self.tool_dir = os.path.join(workspace_dir(), 'tools', 'evaluate_model')
        self.input_path = os.path.join(self.tool_dir, 'input')
        self.output_path = os.path.join(self.tool_dir, 'output')

        if not os.path.exists(self.input_path):
            os.makedirs(self.input_path)
            print(f"Created input directory: {self.input_path}")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")

    def get_data_mscoco(self)->Tuple[List[np.ndarray],List[dict],dict]:

        annotation_path = os.path.join(self.input_path, "ann.json")
        images_dir = os.path.join(self.input_path, "images")

        labels_to_id = dict()
        images = []
        targets = []

        dataset = pycocotools.coco.COCO(annotation_path)

        img_ids = dataset.getImgIds()
        for img_id in img_ids:
            
            anns_ids = dataset.getAnnIds(imgIds=img_id)

            img = dataset.loadImgs(img_id)[0]
            anns = dataset.loadAnns(anns_ids)

            file_name = img["file_name"]
            img_path = os.path.join(images_dir, file_name)

            if not os.path.exists(img_path):
                print(f"Image file {file_name} does not exist, skipping.")
                continue
            image = cv2.imread(img_path)
            h, w, c = image.shape

            boxes = list()
            masks = list()
            labels = list()

            for ann in anns:

                # extract label
                cat_id = ann['category_id']
                cat = dataset.loadCats(cat_id)[0]
                label = cat['name']
                if label not in labels_to_id:
                    labels_to_id[label] = len(labels_to_id)
                label_id = labels_to_id[label]

                # extract bbox
                x1,y1,xl,yl = ann['bbox']
                x2 = x1+xl
                y2 = y1+yl

                # extract mask
                mask = dataset.annToMask(ann)

                # save data
                boxes.append(torch.FloatTensor([x1, y1, x2, y2]).cpu())
                masks.append(torch.BoolTensor(mask).cpu())
                labels.append(label_id)

            # add data to targets
            if len(boxes) == 0:
                print(f"No valid shapes found in {file_name}, skipping.")
                continue
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)
            targets.append({
                'boxes': boxes,
                'masks': masks,
                'labels': torch.IntTensor(labels).cpu()
            })
            images.append(image)



        return images, targets, labels_to_id

    def get_data_label_studio(self)->Tuple[List[np.ndarray],List[dict],dict]:

        annotation_path = os.path.join(self.input_path, "ann.json")
        images_dir = os.path.join(self.input_path, "images")

        labels_to_id = dict()
        images = []
        targets = []

        # Open and read the JSON file
        with open(annotation_path, 'r') as file:
            data: dict = json.load(file)

        for img_data in data:

            #open img
            file_name = img_data["file_upload"]
            img_path = os.path.join(images_dir, file_name)

            if not os.path.exists(img_path):
                print(f"Image file {file_name} does not exist, skipping.")
                continue
            image = cv2.imread(img_path)
            h, w, c = image.shape

            #open annotations
            anns_data:dict = img_data["annotations"][0]["result"]

            #parse annotations
            boxes = list()
            masks = list()
            labels = list()
            for ann_data in anns_data:

                if(ann_data["type"] != "brushlabels"):
                    continue

                # extract label
                label = ann_data["value"]["brushlabels"][0]
                if label not in labels_to_id:
                    labels_to_id[label] = len(labels_to_id)
                label_id = labels_to_id[label]

                

                # extract mask
                rle_mask = ann_data["value"]["rle"]
                mask = rle_to_mask(rle_mask, h, w)

                # extract bbox
                xs = np.any(mask, axis=1)
                ys = np.any(mask, axis=0)
                y1, y2 = np.where(ys)[0][[0, -1]]
                x1, x2 = np.where(xs)[0][[0, -1]]

                # save data
                boxes.append(torch.FloatTensor([x1, y1, x2, y2]).cpu())
                masks.append(torch.BoolTensor(mask).cpu())
                labels.append(label_id)

            # add data to targets
            if len(boxes) == 0:
                print(f"No valid shapes found in {file_name}, skipping.")
                continue
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)
            targets.append({
                'boxes': boxes,
                'masks': masks,
                'labels': torch.IntTensor(labels).cpu()
            })
            images.append(image)

        return images, targets, labels_to_id

    def evaluate_model(self):
        # Load input
        images, targets, labels_to_id = self.get_data_label_studio()

        if len(targets) == 0:
            print("No valid targets found, exiting.")
            return
        
        # setup reference model
        label_str_list = list(labels_to_id.keys())
        self.ref_model.set_classes(label_str_list, self.ref_model.get_text_pe(label_str_list))

        # Load the model learning with all classes
        self.embeddings_generator.update()
        embeddings, alias_name_list, category_alias_to_name, alias_score_exponent_list = self.embeddings_generator.get_embedding(categories_name=None, fallback_model=self.trained_model_with_no_classes_specification)
        if len(embeddings) > 0:
            self.trained_model_with_no_classes_specification.set_classes(alias_name_list, embeddings)

        # Evaluate the model
        print("Evaluating model...")

        trained_results = self.trained_model_with_no_classes_specification.predict(images, conf=0.01, verbose=False)
        ref_results = self.ref_model.predict(images, conf=0.01, verbose=False)

        trained_preds = []
        trained_with_conf_learning_preds = []
        ref_preds = []

        for i in range(len(images)):
            ref_result = ref_results[i]
            trained_result = trained_results[i]

            # trained result extraction
            trained_boxes = trained_result.boxes.xyxy.cpu()
            trained_conf = trained_result.boxes.conf.cpu()
            
            trained_masks = extract_resized_masks(trained_result).cpu()
            trained_labels_nbr = trained_result.boxes.cls.size()[0]
            trained_labels_alias = [alias_name_list[int(trained_result.boxes.cls[j].item())] for j in range(trained_labels_nbr) ]
            trained_labels_names = [category_alias_to_name[trained_labels_alias[j]] for j in range(trained_labels_nbr)]

            trained_conf_corrected = torch.clone(trained_conf)
            trained_conf_corrected = torch.tensor([ x.item() ** alias_score_exponent_list[int(trained_result.boxes.cls[j].item())] for j, x in enumerate(torch.unbind(trained_conf_corrected))])

            valid_ids = [j for j in range(trained_labels_nbr) if trained_labels_names[j] in labels_to_id]
            trained_labels_names = [trained_labels_names[j] for j in valid_ids]
            trained_boxes = trained_boxes[valid_ids]
            trained_masks = trained_masks[valid_ids]
            trained_conf = trained_conf[valid_ids]
            trained_conf_corrected = trained_conf_corrected[valid_ids]
            trained_labels_ids = [labels_to_id[name] for name in trained_labels_names]
            trained_labels_ids_tensor = torch.IntTensor(trained_labels_ids).cpu()

            # Ref result extraction
            ref_boxes = ref_result.boxes.xyxy.cpu()
            ref_conf = ref_result.boxes.conf.cpu()
            ref_masks = extract_resized_masks(ref_result).cpu()
            ref_labels_nbr = ref_result.boxes.cls.size()[0]
            ref_labels_names = [ref_result.names[ref_result.boxes.cls[j].item()] for j in range(ref_labels_nbr) ]
            ref_labels_ids = [labels_to_id[name] for name in ref_labels_names ]
            ref_labels_ids_tensor = torch.IntTensor(ref_labels_ids).cpu()

            # Append predictions
            ref_preds.append({
                'boxes': ref_boxes,
                'masks': ref_masks,
                'labels': ref_labels_ids_tensor,
                'scores': ref_conf
            })

            trained_preds.append({
                'boxes': trained_boxes,
                'masks': trained_masks,
                'labels': trained_labels_ids_tensor,
                'scores': trained_conf
            })

            trained_with_conf_learning_preds.append({
                'boxes': trained_boxes,
                'masks': trained_masks,
                'labels': trained_labels_ids_tensor,
                'scores': trained_conf_corrected
            })

        map_ref_seg = mean_average_precision(
            preds=ref_preds,
            target=targets,
            iou_type="segm",
            class_metrics=True
        )
        map_trained_seg = mean_average_precision(
            preds=trained_preds,
            target=targets,
            iou_type="segm",
            class_metrics=True
        )
        map_trained_with_conf_learning_seg = mean_average_precision(
            preds=trained_with_conf_learning_preds,
            target=targets,
            iou_type="segm",
            class_metrics=True
        )

        #map_ref_bbox = mean_average_precision(
        #    preds=ref_preds,
        #    target=targets,
        #    iou_type="bbox",
        #    class_metrics=True
        #)
        #map_trained_bbox = mean_average_precision(
        #    preds=trained_preds,
        #    target=targets,
        #    iou_type="bbox",
        #    class_metrics=True
        #)
        #map_trained_with_conf_learning_bbox = mean_average_precision(
        #    preds=trained_with_conf_learning_preds,
        #    target=targets,
        #    iou_type="bbox",
        #    class_metrics=True
        #)

        # save the results on disc
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        output_file = os.path.join(self.output_path, 'evaluation_results_'+str(int(time.time()))+'.json')
        with open(output_file, 'w') as f:
            json.dump({
                'Labels ids': labels_to_id,
                'raw model performances' : {
                    "segmentation":
                    {
                        'map': map_ref_seg['map'].item(),
                        'map_50': map_ref_seg['map_50'].item(),
                        'map_75': map_ref_seg['map_75'].item(),
                        'map_small': map_ref_seg['map_small'].item(),
                        'map_medium': map_ref_seg['map_medium'].item(),
                        'map_large': map_ref_seg['map_large'].item(),
                        'map_per_class': map_ref_seg['map_per_class'].cpu().tolist()
                    }#,
                    #"bounding boxes":
                    #{
                    #    'map': map_ref_bbox['map'].item(),
                    #    'map_50': map_ref_bbox['map_50'].item(),
                    #    'map_75': map_ref_bbox['map_75'].item(),
                    #    'map_small': map_ref_bbox['map_small'].item(),
                    #    'map_medium': map_ref_bbox['map_medium'].item(),
                    #    'map_large': map_ref_bbox['map_large'].item(),
                    #    'map_per_class': map_ref_bbox['map_per_class'].cpu().tolist()
                    #}
                },
                'database helped model performances': {
                    "segmentation":
                    {
                        'map': map_trained_seg['map'].item(),
                        'map_50': map_trained_seg['map_50'].item(),
                        'map_75': map_trained_seg['map_75'].item(),
                        'map_small': map_trained_seg['map_small'].item(),
                        'map_medium': map_trained_seg['map_medium'].item(),
                        'map_large': map_trained_seg['map_large'].item(),
                        'map_per_class': map_trained_seg['map_per_class'].cpu().tolist()
                    }#,
                    #"bounding boxes":
                    #{
                    #    'map': map_trained_bbox['map'].item(),
                    #    'map_50': map_trained_bbox['map_50'].item(),
                    #    'map_75': map_trained_bbox['map_75'].item(),
                    #    'map_small': map_trained_bbox['map_small'].item(),
                    #    'map_medium': map_trained_bbox['map_medium'].item(),
                    #    'map_large': map_trained_bbox['map_large'].item(),
                    #    'map_per_class': map_trained_bbox['map_per_class'].cpu().tolist()
                    #}
                },
                'database helped model with confidence learning performances': {
                    "segmentation":
                    {
                        'map': map_trained_with_conf_learning_seg['map'].item(),
                        'map_50': map_trained_with_conf_learning_seg['map_50'].item(),
                        'map_75': map_trained_with_conf_learning_seg['map_75'].item(),
                        'map_small': map_trained_with_conf_learning_seg['map_small'].item(),
                        'map_medium': map_trained_with_conf_learning_seg['map_medium'].item(),
                        'map_large': map_trained_with_conf_learning_seg['map_large'].item(),
                        'map_per_class': map_trained_with_conf_learning_seg['map_per_class'].cpu().tolist()
                    }#,
                    #"bounding boxes":
                    #{
                    #    'map': map_trained_with_conf_learning_bbox['map'].item(),
                    #    'map_50': map_trained_with_conf_learning_bbox['map_50'].item(),
                    #    'map_75': map_trained_with_conf_learning_bbox['map_75'].item(),
                    #    'map_small': map_trained_with_conf_learning_bbox['map_small'].item(),
                    #    'map_medium': map_trained_with_conf_learning_bbox['map_medium'].item(),
                    #    'map_large': map_trained_with_conf_learning_bbox['map_large'].item(),
                    #    'map_per_class': map_trained_with_conf_learning_bbox['map_per_class'].cpu().tolist()
                    #}
                }
            }, f, indent=4)
        print(f"Results saved to {output_file}")




def main(args=None):
    rclpy.init()

    node = TaskNode()
    print("Node ready")

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    node.evaluate_model()

if __name__ == '__main__':
    main()