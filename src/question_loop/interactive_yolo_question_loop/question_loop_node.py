from cv_bridge import CvBridge

import rclpy
from threading import Thread, Lock
from rclpy.node import Node
from interactive_yolo_utils import workspace_dir

from database_service.database_service import DatabaseServices
from tensor_msg_conversion.tensor_msg_conversion import boolTensorToNdArray
from .qt_ui.interface_question import interface_question

from sensor_msgs.msg import Image as RosImage
from interactive_yolo_interfaces.msg import DatabaseQuestionInfo, DatabaseImageInfo, Bbox, Prediction, PredictionResult

import cv2
import time
import os

import numpy as np

import sys
from PySide6 import QtWidgets
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect

class QuestionLoopNode(Node):
    def __init__(self):
        super().__init__('demo_node')

        self.cv_bridge = CvBridge()

        self.image_lock = Lock()
        self.image = None
        self.freeze_image = False

        self.database = DatabaseServices(self)

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        self.model_input_publisher = self.create_publisher(RosImage, 'interactive_yolo/model_input_image', qos_profile=qos_policy)

        self.sub_input_image = self.create_subscription(
            PredictionResult, 
            'interactive_yolo/model_output_predictions',
            self._model_output_update_callback,
            qos_profile=qos_policy)

        self.capture_effect = QSoundEffect()
        self.capture_effect.setSource(QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), 'sounds', 'capture.wav')))
        self.question_interface = interface_question()
        self.question_interface.set_capture_callback(self._capture_callback)

        self.cam_thread = Thread(target=self.camera_thread_loop, daemon=True)
        self.display_raw_thread = Thread(target=self.display_raw_thread_loop, daemon=True)
        self.question_loop_thread = Thread(target=self.question_loop, daemon=True)

        self.cam_thread.start()
        self.display_raw_thread.start()
        self.question_loop_thread.start()

        self.display_counter = 0
        self.annotated_img = None

    def display_raw_thread_loop(self):

        while True:

            img = None
            with self.image_lock:
                if self.image is not None:
                    img = self.image.copy()
        
            if img is not None:

                img_msg = self.cv_bridge.cv2_to_imgmsg(img, 'bgr8')
                self.model_input_publisher.publish(img_msg)

                self.question_interface.set_capture_image_brute(img)
            
            time.sleep(0.1)

    def _model_output_update_callback(self, msg: PredictionResult):
                
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontThickness = 2

        img = self.cv_bridge.imgmsg_to_cv2(msg.image, 'bgr8')

        predictions : list[Prediction] = msg.predictions
        classes_names = dict()
        classes_threshold = dict()
        class_nbr = 0

        for prediction in predictions:
            if prediction.class_name not in classes_names.keys():
                classes_names[prediction.class_name] = class_nbr
                class_nbr += 1

            threshold = prediction.confidence * 0.8
            if prediction.class_name not in classes_threshold.keys():
                classes_threshold[prediction.class_name] = threshold
            elif classes_threshold[prediction.class_name] < threshold:
                classes_threshold[prediction.class_name] = threshold
        
        color_map_coef = 0.5
        if(class_nbr > 1):
            color_map_coef = 1.0 / (class_nbr-1)
        
        for prediction in predictions:
            if prediction.confidence < classes_threshold[prediction.class_name]:
                continue

            color_gray_Value = int(255*classes_names[prediction.class_name]*color_map_coef)
            color = cv2.cvtColor(cv2.applyColorMap(np.uint8([[[color_gray_Value]]]), cv2.COLORMAP_JET),cv2.COLOR_RGB2BGR)[0][0]

            label = prediction.class_name +":"+ f"{prediction.confidence:.2f}"
            label = label.replace("é","e").replace("ê","e").replace("è","e").replace("ë","e")
            label = label.replace("à","a").replace("â","a")
            label = label.replace("û","u")
            label = label.replace("ç","c")
            img = cv2.rectangle(img, (int(prediction.bbox.x1), int(prediction.bbox.y1)), (int(prediction.bbox.x2), int(prediction.bbox.y2)), color.tolist(), 2)
            img = cv2.putText(img, label, (int(prediction.bbox.x1), int(prediction.bbox.y1 + 22)), font, fontScale, color.tolist(), fontThickness, cv2.LINE_AA)

        self.question_interface.set_capture_image_annotee(img)

    def camera_thread_loop(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open camera")

        while True:
            ret, frame = cap.read()

            if not ret:
                raise Exception("Cannot read camera")

            with self.image_lock:
                self.image = frame.copy()

    def question_loop(self):

        while(True):
            print("question sleep")
            time.sleep(5.0)

            print("try get question")
            question_request_answer = self.database.GetDatabaseQuestion()

            if question_request_answer is None:
                continue

            question_info: DatabaseQuestionInfo = question_request_answer.info
            if question_info.id == -1:
                continue

            image_id = question_info.image_id
            if question_info.image_id == -1:
                continue

            image_request_answer = self.database.GetDatabaseImage(image_id)
            if image_request_answer is None:
                continue
            
            image_info : DatabaseImageInfo = image_request_answer.info
            if image_info.id != image_id:
                continue

            cv_image = cv2.imread(image_info.path)

            mask = boolTensorToNdArray(question_info.mask)

            img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            cv_image_alt = img_gray_bgr.copy()
            color = [0,0,255]
            cv_image_alt[mask] = color

            alpha = 0.25
            beta = 1.0 - alpha

            cv_image = cv2.addWeighted(img_gray_bgr, alpha, cv_image_alt, beta, 0)

            print("ask question")
            start_ask_time = time.time()
            category_name = self.question_interface.ask_question(cv_image)
            interaction_time = time.time() - start_ask_time

            print("solve question")
            self.database.SolveDatabaseQuestion(question_info.id, category_name, interaction_time)

    def _capture_callback(self, cv_image_1, cv_image_2, use_img_2):

        image_to_save = cv_image_2 if use_img_2 else cv_image_1

        filename = "image_"+str(int(time.time()))+".png"
        folder = os.path.join(workspace_dir(),"tools","captured_images")

        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created capture dir: {folder}")

        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, image_to_save)

        self.capture_effect.setLoopCount(1)
        self.capture_effect.setVolume(0.5)
        self.capture_effect.play()

def main(args=None):
    app = QtWidgets.QApplication([])
    rclpy.init()

    node = QuestionLoopNode()

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()