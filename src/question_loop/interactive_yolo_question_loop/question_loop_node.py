from cv_bridge import CvBridge

import rclpy
from threading import Thread, Lock
from rclpy.node import Node
from interactive_yolo_utils import workspace_dir
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
from playsound import playsound

class QuestionLoopNode(Node):

    def __init__(self):
        super().__init__('demo_node')
        self.declare_parameter('image_sending_mode', 'raw')
        self.declare_parameter('input_mode', 'pc')
        self.declare_parameter('rescale_question', False)

        self.compressed_image = (self.get_parameter('image_sending_mode').value == 'compressed')
        self.ttop_input = (self.get_parameter('input_mode').value == 'ttop')
        self.rescale_question = self.get_parameter('rescale_question').value

        self.cv_bridge = CvBridge()

        self.image_lock = Lock()
        self.image = None
        self.freeze_image = False

        self.database = DatabaseServices(self)

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        if self.compressed_image:
            self.model_input_publisher = self.create_publisher(RosCompressedImage, 'interactive_yolo/model_input_image_compressed', qos_profile=qos_policy)
        else:
            self.model_input_publisher = self.create_publisher(RosImage, 'interactive_yolo/model_input_image', qos_profile=qos_policy)

        self.speak_listen = SpeakListen(self)

        self.sub_input_image = self.create_subscription(
            PredictionResult, 
            'interactive_yolo/model_output_predictions',
            self._model_output_update_callback,
            qos_profile=qos_policy)

        self.capture_sound_file = os.path.join(workspace_dir(),"src","question_loop","interactive_yolo_question_loop", 'sounds', 'capture.wav')
        self.question_interface = interface_question()
        self.question_interface.set_capture_callback(self._capture_callback)

        if self.ttop_input:
            self.ttop_camera_input_subscriber = self.create_subscription( RosImage, 'interactive_yolo/ttop_camera_input', self._ttop_camera_input_callback, qos_profile=qos_policy)
        else:
            self.cam_thread = Thread(target=self.camera_thread_loop, daemon=True)
            self.cam_thread.start()

        self.display_raw_thread = Thread(target=self.display_raw_thread_loop, daemon=True)
        self.question_loop_thread = Thread(target=self.question_loop, daemon=True)

        self.display_raw_thread.start()
        self.question_loop_thread.start()

        self.display_counter = 0
        self.annotated_img = None

        self._person_score = 0.0
        self._person_score_lock = Lock()

    def register_best_person_score(self, score:float, alpha:float = 0.5):
        with self._person_score_lock:
            self._person_score = (self._person_score * alpha) + (score * (1.0 - alpha))

    def person_score(self) -> float:
        with self._person_score_lock:
            return self._person_score
    
    def display_raw_thread_loop(self):

        while True:

            img = None
            with self.image_lock:
                if self.image is not None:
                    img = self.image.copy()
        
            if img is not None:
                if self.compressed_image:
                    img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(img, 'jpg')
                    self.model_input_publisher.publish(img_msg)
                else:
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
        best_person_score = 0.0

        for prediction in predictions:

            if prediction.class_name == "personne":
                if prediction.confidence > best_person_score:
                    best_person_score = prediction.confidence
        
            if prediction.class_name not in classes_names.keys():
                classes_names[prediction.class_name] = class_nbr
                class_nbr += 1

            threshold = prediction.confidence * 0.8
            if prediction.class_name not in classes_threshold.keys():
                classes_threshold[prediction.class_name] = threshold
            elif classes_threshold[prediction.class_name] < threshold:
                classes_threshold[prediction.class_name] = threshold
        
        self.register_best_person_score(best_person_score)
        
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
            
            self.register_best_person_score(0.0, 0.99)
    
    def _ttop_camera_input_callback(self, msg):
        with self.image_lock:
            self.image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.register_best_person_score(0.0, 0.99)

    def question_loop(self):

        while(True):
            print("question sleep")
            time.sleep(5.0)

            if( self.person_score() < 0.2):
                print("no person detected, skip question")
                continue

            print("try get question")
            question_request_answer = self.database.GetDatabaseQuestion()

            if question_request_answer is None:
                continue

            question_info: DatabaseQuestionInfo = question_request_answer.info
            question_score: float = question_request_answer.score
            if question_info.id == -1:
                continue

            if question_score < 0.1:
                continue

            image_id = question_info.image_id
            if question_info.image_id == -1:
                continue

            image_request_answer = self.database.OpenDatabaseImage(image_id)
            if image_request_answer is None:
                continue
            
            image_msg = image_request_answer.image
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)

            mask = boolTensorToNdArray(question_info.mask)

            img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            img_bgr_mask = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            img_gray = img_gray.copy()
            img_gray[:] = [0]
            img_gray[mask] = [255]

            _, thresh = cv2.threshold(img_gray, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            color_a = [0,0,0]
            color_b = [0,0,255]

            cv2.drawContours(img_bgr_mask, contours, -1, color_a, 11)
            cv2.drawContours(img_bgr_mask, contours, -1, color_b, 7)
            cv2.drawContours(img_bgr_mask, contours, -1, color_a, 3)
            img_bgr_mask[mask] = cv_image[mask]

            # get mask bbox
            if self.rescale_question:
                xs = np.any(mask, axis=1)
                ys = np.any(mask, axis=0)
                y1, y2 = np.where(ys)[0][[0, -1]]
                x1, x2 = np.where(xs)[0][[0, -1]]

                y1 = max(y1-100, 0)
                y2 = min(y2+100, img_bgr_mask.shape[1]-1)
                x1 = max(x1-100, 0)
                x2 = min(x2+100, img_bgr_mask.shape[0]-1)

                img_bgr_mask = img_bgr_mask[x1:x2, y1:y2, :]

            estimation_label = None
            if question_info.estimation_category_id >= 0:
                estimation_category_response = self.database.GetDatabaseCategory(question_info.estimation_category_id)
                if estimation_category_response is not None:
                    if estimation_category_response.info.id == question_info.estimation_category_id:
                        estimation_label = estimation_category_response.info.name


            print("ask question")
            self.question_interface.question_widget.setQuestion("Quel est cet objet? score = "+str(question_score))
            start_ask_time = time.time()
            category_name, success = self.question_interface.ask_question(img_bgr_mask, self.speak_listen, estimation_label)
            interaction_time = time.time() - start_ask_time

            print("solve question")
            if success:
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

        playsound(self.capture_sound_file, block=False)

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