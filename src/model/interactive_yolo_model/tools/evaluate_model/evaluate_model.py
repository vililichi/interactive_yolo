import os

from torchmetrics.functional.detection.map import mean_average_precision

import rclpy
from rclpy.node import Node
from .....common.python.database_service import DatabaseServices
from ...utils.embedding_generator import EmbeddingGenerator
from ...utils.model_loader.base_model import base_model
from ...utils.new_object_detection_helper import extract_resized_masks
from interactive_yolo_utils import workspace_dir
from threading import Thread

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time

import cv2

import base64
from io import BytesIO
from PIL import Image
import numpy as np



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

    def evaluate_model(self):
        # Load input
        input_files = [f for f in os.listdir(self.input_path) if f.endswith('.json')]
        
        labels_to_id = dict()
        targets = list()
        images = list()

        for input_file in input_files:
            input_file_path = os.path.join(self.input_path, input_file)
            print(f"Processing input file: {input_file_path}")

            # Load the input data
            with open(input_file_path, 'r') as f:
                input_data = json.load(f)

            image_file = input_data['imagePath']
            image_path = os.path.join(self.input_path, image_file)
            if not os.path.exists(image_path):
                print(f"Image file {image_file} does not exist, skipping.")
                continue
            image = cv2.imread(image_path)
            w = input_data["imageWidth"]
            h = input_data["imageHeight"]

            boxes = list()
            masks = list()
            labels = list()

            for shape in input_data['shapes']:

                # extract label
                label = shape['label']
                if label not in labels_to_id:
                    labels_to_id[label] = len(labels_to_id)
                label_id = labels_to_id[label]

                # extract mask
                image_bytes = base64.b64decode(shape['mask'])
                mask_image = Image.open(BytesIO(image_bytes))
                mask_located = np.array(mask_image)
                mask_located = mask_located.astype(bool)

                x1,y1 = shape['points'][0]
                x2,y2 = shape['points'][1]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                mask: np.ndarray = np.zeros((h, w), dtype=bool)
                mask[y1:(y2+1), x1:(x2+1)] = mask_located

                # save data
                boxes.append(torch.FloatTensor([x1, y1, x2, y2]).cpu())
                masks.append(torch.BoolTensor(mask).cpu())
                labels.append(label_id)
    
            # add data to targets
            if len(boxes) == 0:
                print(f"No valid shapes found in {input_file}, skipping.")
                continue
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)
            targets.append({
                'boxes': boxes,
                'masks': masks,
                'labels': torch.IntTensor(labels).cpu()
            })
            images.append(image)

        if len(targets) == 0:
            print("No valid targets found, exiting.")
            return
        
        # setup reference model
        label_str_list = list(labels_to_id.keys())
        self.ref_model.set_classes(label_str_list, self.ref_model.get_text_pe(label_str_list))

        # Load the model learning
        self.embeddings_generator.update()
        embeddings, alias_name_list, category_alias_to_name = self.embeddings_generator.get_embedding(categories_name=label_str_list, fallback_model=self.trained_model)

        if len(embeddings) > 0:
            self.trained_model.set_classes(alias_name_list, embeddings)
        
        # Load the model learning with all classes
        ncs_embeddings, ncs_alias_name_list, ncs_category_alias_to_name = self.embeddings_generator.get_embedding(categories_name=[], fallback_model=self.trained_model_with_no_classes_specification)
        if len(ncs_embeddings) > 0:
            self.trained_model_with_no_classes_specification.set_classes(ncs_alias_name_list, ncs_embeddings)

        # Evaluate the model
        print("Evaluating model...")

        trained_results = self.trained_model.predict(images, conf=0.05, verbose=False)
        trained_ncs_results = self.trained_model_with_no_classes_specification.predict(images, conf=0.05, verbose=False)
        ref_results = self.ref_model.predict(images, conf=0.05, verbose=False)

        trained_preds = []
        trained_ncs_preds = []
        ref_preds = []

        for i in range(len(images)):
            trained_result = trained_results[i]
            ref_result = ref_results[i]

            # Convert results to the format expected by MeanAveragePrecision
            trained_boxes = trained_result.boxes.xyxy.cpu()
            trained_conf = trained_result.boxes.conf.cpu()
            trained_masks = extract_resized_masks(trained_result).cpu()
            trained_labels_nbr = trained_result.boxes.cls.size()[0]
            trained_labels_alias = [trained_result.names[trained_result.boxes.cls[i].item()] for i in range(trained_labels_nbr) ]
            trained_labels_names = [category_alias_to_name[trained_labels_alias[i]] for i in range(trained_labels_nbr)]
            trained_labels_ids = [labels_to_id[name] for name in trained_labels_names]
            trained_labels_ids_tensor = torch.IntTensor(trained_labels_ids).cpu()

            trained_ncs_boxes = trained_ncs_results[i].boxes.xyxy.cpu()
            trained_ncs_conf = trained_ncs_results[i].boxes.conf.cpu()
            trained_ncs_masks = extract_resized_masks(trained_ncs_results[i]).cpu()
            trained_ncs_labels_nbr = trained_ncs_results[i].boxes.cls.size()[0]
            trained_ncs_labels_alias = [trained_ncs_results[i].names[trained_ncs_results[i].boxes.cls[j].item()] for j in range(trained_ncs_labels_nbr) ]
            trained_ncs_labels_names = [ncs_category_alias_to_name[trained_ncs_labels_alias[j]] for j in range(trained_ncs_labels_nbr)]

            valid_ids = [i for i in range(trained_ncs_labels_nbr) if trained_ncs_labels_names[i] in labels_to_id]
            trained_ncs_labels_names = [trained_ncs_labels_names[i] for i in valid_ids]
            trained_ncs_boxes = trained_ncs_boxes[valid_ids]
            trained_ncs_masks = trained_ncs_masks[valid_ids]
            trained_ncs_conf = trained_ncs_conf[valid_ids]

            trained_ncs_labels_ids = [labels_to_id[name] for name in trained_ncs_labels_names]
            trained_ncs_labels_ids_tensor = torch.IntTensor(trained_ncs_labels_ids).cpu()

            ref_boxes = ref_result.boxes.xyxy.cpu()
            ref_conf = ref_result.boxes.conf.cpu()
            ref_masks = extract_resized_masks(ref_result).cpu()
            ref_labels_nbr = ref_result.boxes.cls.size()[0]
            ref_labels_names = [ref_result.names[ref_result.boxes.cls[i].item()] for i in range(ref_labels_nbr) ]
            ref_labels_ids = [labels_to_id[ref_labels_names[i]] for i in range(ref_labels_nbr) ]
            ref_labels_ids_tensor = torch.IntTensor(ref_labels_ids).cpu()

            # Append predictions
            trained_preds.append({
                'boxes': trained_boxes,
                'masks': trained_masks,
                'labels': trained_labels_ids_tensor,
                'scores': trained_conf
            })

            ref_preds.append({
                'boxes': ref_boxes,
                'masks': ref_masks,
                'labels': ref_labels_ids_tensor,
                'scores': ref_conf
            })

            trained_ncs_preds.append({
                'boxes': trained_ncs_boxes,
                'masks': trained_ncs_masks,
                'labels': trained_ncs_labels_ids_tensor,
                'scores': trained_ncs_conf
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
        map_trained_ncs_seg = mean_average_precision(
            preds=trained_ncs_preds,
            target=targets,
            iou_type="segm",
            class_metrics=True
        )

        map_ref_bbox = mean_average_precision(
            preds=ref_preds,
            target=targets,
            iou_type="bbox",
            class_metrics=True
        )
        map_trained_bbox = mean_average_precision(
            preds=trained_preds,
            target=targets,
            iou_type="bbox",
            class_metrics=True
        )
        map_trained_ncs_bbox = mean_average_precision(
            preds=trained_ncs_preds,
            target=targets,
            iou_type="bbox",
            class_metrics=True
        )

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
                    },
                    "bounding boxes":
                    {
                        'map': map_ref_bbox['map'].item(),
                        'map_50': map_ref_bbox['map_50'].item(),
                        'map_75': map_ref_bbox['map_75'].item(),
                        'map_small': map_ref_bbox['map_small'].item(),
                        'map_medium': map_ref_bbox['map_medium'].item(),
                        'map_large': map_ref_bbox['map_large'].item(),
                        'map_per_class': map_ref_bbox['map_per_class'].cpu().tolist()
                    }
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
                    },
                    "bounding boxes":
                    {
                        'map': map_trained_bbox['map'].item(),
                        'map_50': map_trained_bbox['map_50'].item(),
                        'map_75': map_trained_bbox['map_75'].item(),
                        'map_small': map_trained_bbox['map_small'].item(),
                        'map_medium': map_trained_bbox['map_medium'].item(),
                        'map_large': map_trained_bbox['map_large'].item(),
                        'map_per_class': map_trained_bbox['map_per_class'].cpu().tolist()
                    }
                },
                'database helped model with no classes specification performances': {
                    "segmentation":
                    {
                        'map': map_trained_ncs_seg['map'].item(),
                        'map_50': map_trained_ncs_seg['map_50'].item(),
                        'map_75': map_trained_ncs_seg['map_75'].item(),
                        'map_small': map_trained_ncs_seg['map_small'].item(),
                        'map_medium': map_trained_ncs_seg['map_medium'].item(),
                        'map_large': map_trained_ncs_seg['map_large'].item(),
                        'map_per_class': map_trained_ncs_seg['map_per_class'].cpu().tolist()
                    },
                    "bounding boxes":
                    {
                        'map': map_trained_ncs_bbox['map'].item(),
                        'map_50': map_trained_ncs_bbox['map_50'].item(),
                        'map_75': map_trained_ncs_bbox['map_75'].item(),
                        'map_small': map_trained_ncs_bbox['map_small'].item(),
                        'map_medium': map_trained_ncs_bbox['map_medium'].item(),
                        'map_large': map_trained_ncs_bbox['map_large'].item(),
                        'map_per_class': map_trained_ncs_bbox['map_per_class'].cpu().tolist()
                    }
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