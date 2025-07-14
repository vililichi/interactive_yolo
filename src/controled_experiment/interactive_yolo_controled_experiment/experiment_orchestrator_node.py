from cv_bridge import CvBridge

import rclpy
from threading import Thread, Lock
from rclpy.node import Node
from interactive_yolo_utils import workspace_dir, FreqMonitor
from speaklisten import SpeakListen

from database_service.database_service import DatabaseServices
from tensor_msg_conversion.tensor_msg_conversion import boolTensorToNdArray
from .qt_ui.interface_question import interface_question

from sensor_msgs.msg import Image as RosImage, CompressedImage as RosCompressedImage
from interactive_yolo_interfaces.msg import DatabaseQuestionInfo, DatabaseImageInfo, Bbox, Prediction, PredictionResult

import cv2
import time
import os

import numpy as np

import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl

playsoud_available = True
try:
    from playsound import playsound
except:
    playsoud_available = False

class Experiment_node(Node):

    def __init__(self):
        super().__init__('experiment_orchestrator_node')

def main(args=None):

    rclpy.init()

    node = Experiment_node()

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()