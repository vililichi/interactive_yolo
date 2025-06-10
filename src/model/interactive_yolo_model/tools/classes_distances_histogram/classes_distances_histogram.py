import os

from rclpy.executors import MultiThreadedExecutor

import rclpy
from rclpy.node import Node
from database_service import DatabaseServices
from tensor_msg_conversion import float32TensorToTorchTensor
from interactive_yolo_utils import workspace_dir
from threading import Thread

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

class TaskNode(Node):
    def __init__(self):
        super().__init__('embedding_distance_histogram_node')

        self.database_services = DatabaseServices(self)

        self.tool_dir = os.path.join(workspace_dir(), 'tools', 'classes_distances_histogram')
        self.output_path = os.path.join(self.tool_dir, 'output')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")

        self.time_str = str(int(time.time()))

    def generate_histogram(self):
        
        annotations_infos = list()
        logs = list()
        cat_infos = self.database_services.GetAllDatabaseCategories().infos

        # Get nothing class id
        nothing_class_id = self.database_services.GetDatabaseCategoryByName("__NOTHING__").info.id

        # Get all annotations infos from the database
        for cat_info in cat_infos:

            annotations_ids = cat_info.annotations_ids

            if len(annotations_ids) == 0:
                continue

            for annotation_id in annotations_ids:
                annotation_info = self.database_services.GetDatabaseAnnotation(annotation_id).info
                annotations_infos.append(annotation_info)

        # Generate the histogram
        interclasses_dist = list()
        intraclasses_dist = list()
        for annotation_info_a in annotations_infos:

            embedding_a = float32TensorToTorchTensor(annotation_info_a.embedding)
            if embedding_a is None:
                continue

            id_a = annotation_info_a.id
            class_a = annotation_info_a.category_id

            for annotation_info_b in annotations_infos:

                embedding_b = float32TensorToTorchTensor(annotation_info_b.embedding)
                if embedding_b is None:
                    continue
                
                id_b = annotation_info_b.id
                class_b = annotation_info_b.category_id

                if class_a == nothing_class_id and class_b == nothing_class_id:
                    continue

                if id_a <= id_b:
                    continue

                dist = torch.dist(embedding_a, embedding_b).item()

                log = "ids : " + str(id_a)+"-" + str(id_b) +" | classes_ids : "+str(class_a)+"-"+str(class_b)+" | dist : " + str(dist)

                if class_a == class_b:
                    intraclasses_dist.append(dist)
                    log += " | intra-class"
                else:
                    interclasses_dist.append(dist)
                    log += " | inter-class"

                logs.append(log)

        # Print logs
        for log in logs:
            print(log)
        
        # save logs in a text file
        
        log_file_name = 'embedding_distance_histogram_logs_'+self.time_str+'.txt'
        log_file_path = os.path.join(self.output_path, log_file_name)
        with open(log_file_path, 'w') as f:
            for log in logs:
                f.write(log + '\n')
        print(f"Logs saved to {log_file_path}")

        # Convert to numpy arrays
        interclasses_dist = np.array(interclasses_dist)
        intraclasses_dist = np.array(intraclasses_dist)

        # create bins for the histogram
        nbr_bins = 30
        bins = np.linspace(0, max(np.max(interclasses_dist), np.max(intraclasses_dist)), nbr_bins)
        
        # Plot the histogram
        plt.figure(figsize=(10, 5))
        plt.hist(interclasses_dist, alpha=0.5, density=True, bins=bins, label='Inter-classes', color='blue')
        plt.hist(intraclasses_dist, alpha=0.5, density=True, bins=bins, label='Intra-classes', color='orange')
        plt.title('Embedding Distance Histogram')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend(['Inter-classes', 'Intra-classes'])
        plt.tight_layout()
        # Save the histogram to a file
        fig_file_name = 'embedding_distance_histogram_'+self.time_str+'.png'
        fig_file_path = os.path.join(self.output_path, fig_file_name)
        plt.savefig(fig_file_path)
        print(f"Histogram saved to {fig_file_path}")


def main(args=None):
    rclpy.init()

    node = TaskNode()
    print("Node ready")

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    node.generate_histogram()

if __name__ == '__main__':
    main()