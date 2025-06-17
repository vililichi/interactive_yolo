import os
os.environ['YOLO_VERBOSE'] = 'False'

from .utils.model_loader.base_model import base_model, sam_model, fast_sam_model
from .utils.model_loader.fast_model import generate_fast_model
from .utils.new_object_detection_helper import new_object_detection, NewObjectDetectionParameters
from .utils.embedding_generator import EmbeddingGenerator
from tensor_msg_conversion import float32TensorToTorchTensor, boolTensorToNdArray
import time
from typing import List, Tuple, Any, Dict, Union
from threading import Thread, Lock
from queue import Queue
from interactive_yolo_interfaces.msg import DatabaseUpdateNotifaction, DatabaseCategoryInfo, PredictionResult, Prediction, Bbox
from sensor_msgs.msg import Image as RosImage

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from database_service.database_service import DatabaseServices

from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

import torch
import numpy as np

from queue import PriorityQueue

import cv2

class ExplorationNode(Node):

    def __init__(self):
        super().__init__('learning_node')

        # models
        self.embedding_generation_model = base_model()
        self.embedding_generation_model_lock = Lock()

        self.prototypal_model = base_model()
        self.prototypal_model_lock = Lock()

        self.working_model = None
        self.working_model_lock = Lock()
        self.embeddings, self.alias_name_list, self.category_alias_to_name = None, None, None

        self.sam_model = fast_sam_model()
        self.sam_model_lock = Lock()

        # update working model
        self.check_model_update = True
        self.task_thread_update_working_model = Thread(target=self._task_loop_update_working_model, daemon=True)

        # image to annotate
        self.input_lock = Lock()
        self.input_update_time = 0.0
        self.input_image = None

        # annotated image
        self.output_lock = Lock()
        self.output_update_time = 0.0
        self.output_image = None
        self.output_result = None

        # services
        self.cv_bridge = CvBridge()

        self.database_services = DatabaseServices(self)
        self.embeddings_generator = EmbeddingGenerator(self.database_services)

        # Category embedding generation task attributs
        self.task_queue_category_embedding_generation = Queue()
        self.task_thread_category_embedding_generation = Thread(target=self._task_loop_category_embedding_generation, daemon=True)
        self.category_embedding_max_clusterring_error = 0.5

        # Annotation embedding generation task attributs
        self.task_queue_annotation_embedding_generation = Queue()
        self.task_thread_annotation_embedding_generation = Thread(target=self._task_loop_annotation_embedding_generation, daemon=True)

        # scan task attibuts
        self.task_thread_scan = Thread(target=self._task_loop_scan, daemon=True)

        # inference task attributs
        self.inference_dt = 0.2
        self.task_thread_inference = Thread(target=self._task_loop_inference, daemon=True)

        # detect new object task attibuts
        self.task_thread_detect_new_object = Thread(target=self._task_loop_detect_new_object, daemon=True)

        # other
        self.sub_annotation_update = self.create_subscription(
            DatabaseUpdateNotifaction,
            'interactive_yolo/database_annotation_update_notification',
            self._annotation_update_callback,
            10)

        self.sub_category_update = self.create_subscription(
            DatabaseUpdateNotifaction,
            'interactive_yolo/database_category_update_notification',
            self._category_update_callback,
            10)
        
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        
        self.sub_input_image = self.create_subscription(
            RosImage,
            'interactive_yolo/model_input_image',
            self._image_input_update_callback,
            qos_profile=qos_policy)

        self.pub_debug_img_raw = self.create_publisher(RosImage, 'interactive_yolo/new_object_detection_debug_raw', qos_profile=qos_policy)
        self.pub_debug_img_sam = self.create_publisher(RosImage, 'interactive_yolo/new_object_detection_debug_sam', qos_profile=qos_policy)
        self.pub_debug_img_yolo = self.create_publisher(RosImage, 'interactive_yolo/new_object_detection_debug_yolo', qos_profile=qos_policy)
        self.pub_debug_img_unexplained = self.create_publisher(RosImage, 'interactive_yolo/new_object_detection_debug_unexplained', qos_profile=qos_policy)

        self.pub_predictions = self.create_publisher(PredictionResult, 'interactive_yolo/model_output_predictions', qos_profile=qos_policy)

        # start threads
        self.task_thread_scan.start()
        self.task_thread_update_working_model.start()
        self.task_thread_detect_new_object.start()
        self.task_thread_category_embedding_generation.start()
        self.task_thread_annotation_embedding_generation.start()
        self.task_thread_inference.start()

    def _annotation_update_callback(self, msg:DatabaseUpdateNotifaction):
        if msg.type == DatabaseUpdateNotifaction.REGISTERED:
            self.task_queue_annotation_embedding_generation.put(msg.id)

    def _category_update_callback(self, msg:DatabaseUpdateNotifaction):
        if msg.type == DatabaseUpdateNotifaction.REGISTERED:
            self.task_queue_category_embedding_generation.put(time.time())

    def _image_input_update_callback(self, msg:RosImage):
        with self.input_lock:
            self.input_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.input_update_time = time.time()

    def _update_working_model(self):
        if( self.embeddings_generator.update() ):
            with self.embedding_generation_model_lock:
                embeddings, alias_name_list, category_alias_to_name, alias_score_exponent_list = self.embeddings_generator.get_embedding(fallback_model=self.embedding_generation_model)

            if len(embeddings) > 0:

                with self.input_lock:
                    exemple_img = self.input_image.copy()

                with self.prototypal_model_lock:
                    self.prototypal_model.set_classes(alias_name_list, embeddings)
                    fast_model = generate_fast_model(self.prototypal_model, exemple_img)

                with self.working_model_lock:
                    self.working_model = fast_model
                    self.embeddings = embeddings
                    self.alias_name_list = alias_name_list
                    self.category_alias_to_name = category_alias_to_name
                    self.alias_score_exponent_list = alias_score_exponent_list

    def _task_loop_annotation_embedding_generation(self):
        while True:

            # get task
            annotation_id:int = self.task_queue_annotation_embedding_generation.get()
            if annotation_id is None:
                continue
            print("Task loop for annotation embedding generation started for annotation id:", annotation_id)

            # Récupération de l'annotation
            annotation_info = self.database_services.GetDatabaseAnnotation(annotation_id).info

            if annotation_info is None:
                print(" Service GetDatabaseAnnotation non disponible")
                continue

            if annotation_info.id == -1:
                print(" Annotation ",annotation_id," non disponible")
                continue
            
            if annotation_info.id != annotation_id:
                print(" Annotation ",annotation_info.id," reçue au lieu de ",annotation_id)
                continue

            bbox = [ annotation_info.bbox.x1, annotation_info.bbox.y1, annotation_info.bbox.x2, annotation_info.bbox.y2 ]

            # Récupération de l'image
            img_info = self.database_services.GetDatabaseImage(annotation_info.image_id).info
            
            if img_info is None:
                print(" Service GetDatabaseImage non disponible")
                continue

            if img_info.id == -1:
                print(" Image ",annotation_info.image_id," non disponible")
                continue

            if img_info.id != annotation_info.image_id: 
                print(" Image ",img_info.id," reçue au lieu de ",annotation_info.image_id)
                continue
            
            # Génération de l'embedding
            visual_prompts = dict(
                bboxes=[bbox,],
                cls=[0,]
            )
            use_mask = False

            if( annotation_info.mask_set_time > 0.1):
                mask = boolTensorToNdArray(annotation_info.mask).astype(np.float32)
                visual_prompts = dict(
                    masks=[mask,],
                    cls=[0,]
                )
                use_mask = True

            with self.embedding_generation_model_lock:
                vpe = self.embedding_generation_model.generate_vpe(refer_image=img_info.path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, verbose=False)
            
            # Envoie de l'embedding
            self.database_services.SetDatabaseAnnotationEmbedding(annotation_info.id, vpe, use_mask)

            # Ajout d'une tâche pour la génération de l'embedding de la catégorie
            self.task_queue_category_embedding_generation.put(time.time())

            print(" Annotation embedding generation finished")

    def _task_loop_category_embedding_generation(self):

        while True:

            # get task
            update_request_time:float = self.task_queue_category_embedding_generation.get()

            cat_info_list_request = self.database_services.GetAllDatabaseCategories()

            if cat_info_list_request is None:
                continue
            
            cat_info_list : List[DatabaseCategoryInfo] = cat_info_list_request.infos

            for cat_info in cat_info_list:

                category_id = cat_info.id
                if category_id == -1:
                    print(" Categorie ",category_id," non disponible")
                    continue

                print("Task loop for category embedding generation started for category id:", category_id)
                
                if cat_info is None:
                    print(" Service GetDatabaseCategoryByName non disponible")
                    continue
                if cat_info.id == -1:
                    print(" Categorie ",category_id," non disponible")
                    continue
                if cat_info.id != category_id:   
                    print(" Categorie ",cat_info.id," reçue au lieu de ",category_id)
                    continue

                # check if category embedding is already generated
                if cat_info.embeddings_set_time > update_request_time:
                    print(" Category embedding already generated for category id:", cat_info.id)
                    continue

                # text embedding
                tpe = None
                if cat_info.zeroshot_embedding is None:
                    with self.embedding_generation_model_lock:
                        tpe = self.embedding_generation_model.get_text_pe([cat_info.name,]).cpu()
                    self.database_services.SetDatabaseCategoryZeroshotEmbedding(cat_info.id, tpe)                                                                                
                elif cat_info.zeroshot_embedding_set_time < 0.1:
                    with self.embedding_generation_model_lock:
                        tpe = self.embedding_generation_model.get_text_pe([cat_info.name,]).cpu()
                    self.database_services.SetDatabaseCategoryZeroshotEmbedding(cat_info.id, tpe)
                else:
                    tpe = float32TensorToTorchTensor(cat_info.zeroshot_embedding)


                # add visual embedding
                e_clusters_list = [tpe,]
                n_clusters_list = [1,]
                anti_n_n_clusters_list = [1,]
                for id in cat_info.annotations_ids:
                    annotation_info = self.database_services.GetDatabaseAnnotation(id).info

                    if annotation_info is None:
                        print(" Service GetDatabaseAnnotation non disponible")
                        continue

                    if annotation_info.id == -1:
                        print(" Annotation ",id," non disponible")
                        continue

                    if annotation_info.id != id:
                        print(" Annotation ",annotation_info.id," reçue au lieu de ",id)
                        continue

                    if annotation_info.embedding is None:
                        continue

                    if len(annotation_info.embedding.shape) == 0:
                        continue
                    
                    vpe = float32TensorToTorchTensor(annotation_info.embedding)
                    
                    best_error = self.category_embedding_max_clusterring_error
                    best_id = -1
                    for i in range(len(e_clusters_list)):
                        error = torch.dist(e_clusters_list[i], vpe, p=2).item()
                        if error < best_error:
                            best_error = error
                            best_id = i
                    
                    if best_id == -1:
                        e_clusters_list.append(vpe)
                        n_clusters_list.append(1)
                        anti_n_n_clusters_list.append(1)
                    else:

                        old_n = n_clusters_list[best_id]
                        new_n = old_n + 1
                        n_clusters_list[best_id] = new_n

                        e_clusters_list[best_id] = (e_clusters_list[best_id]*old_n + vpe) / new_n

                # Comparaison avec les autres objets

                for alternative_cat_info in cat_info_list:

                    if alternative_cat_info.id == cat_info.id:
                        continue

                    for id in alternative_cat_info.annotations_ids:
                        annotation_info = self.database_services.GetDatabaseAnnotation(id).info

                        if annotation_info is None:
                            print(" Service GetDatabaseAnnotation non disponible")
                            continue

                        if annotation_info.id == -1:
                            print(" Annotation ",id," non disponible")
                            continue

                        if annotation_info.id != id:
                            print(" Annotation ",annotation_info.id," reçue au lieu de ",id)
                            continue

                        if annotation_info.embedding is None:
                            continue

                        if len(annotation_info.embedding.shape) == 0:
                            continue
                        
                        vpe = float32TensorToTorchTensor(annotation_info.embedding)
                        
                        best_error = self.category_embedding_max_clusterring_error
                        best_id = -1
                        for i in range(len(e_clusters_list)):
                            error = torch.dist(e_clusters_list[i], vpe, p=2).item()
                            if error < best_error:
                                best_error = error
                                best_id = i
                        
                        if best_id > -1:
                            anti_n_n_clusters_list[best_id] += 1
                
                if cat_info_list_request is None:
                    score_cluster_list = [1.0 for x in range(len(e_clusters_list))]
                else:
                    score_cluster_list = [ float(anti_n_n_clusters_list[i]) / float(n_clusters_list[i]) for i in range(len(e_clusters_list)) ]

                # Envoie de l'embedding
                self.database_services.SetDatabaseCategoryEmbeddings(cat_info.id, e_clusters_list, score_cluster_list)
                self.check_model_update = True

            print(" Category embedding generation finished")

    def _task_loop_scan(self):
        first_scan = True
        while True:

            # sleep
            if not first_scan:
                time.sleep(60.0)
            else:
                first_scan = False
            print("Task loop for scan started")

            cat_info_list_request = self.database_services.GetAllDatabaseCategories()

            if cat_info_list_request is None:
                return

            cat_info_list : List[DatabaseCategoryInfo] = cat_info_list_request.infos
            
            # check all cats
            personne_cat_exist = False
            for cat_info in cat_info_list:
                if cat_info.id == -1:
                    print(" Categorie ",cat_info.id," non disponible")
                    print(cat_info)
                    continue
                
                # check if the cat is "personne"
                if( cat_info.name == "personne"):
                    personne_cat_exist = True

                # Check if zero shot embedding is generated
                if cat_info.zeroshot_embedding_set_time < 0.1:
                    self.task_queue_category_embedding_generation.put(time.time())

                # var to save if fusion is needed
                embeddings_time = cat_info.embeddings_set_time
                need_embedding_fusion = cat_info.zeroshot_embedding_set_time > embeddings_time

                # Scan all annotations of the category
                for id in cat_info.annotations_ids:

                    # Get annotation
                    annotation_info = self.database_services.GetDatabaseAnnotation(id).info

                    if annotation_info is None:
                        print(" Service GetDatabaseAnnotation non disponible")
                        continue

                    if annotation_info.id == -1:
                        print(" Annotation ",id," non disponible")
                        print(annotation_info)
                        continue

                    if annotation_info.id != id:
                        print(" Annotation ",annotation_info.id," reçue au lieu de ",id)
                        continue

                    # Check if embedding is generated
                    if annotation_info.embedding_set_time < 0.1:
                        self.task_queue_annotation_embedding_generation.put(id)
                    elif annotation_info.mask_set_time > 0.1 and annotation_info.embedding_use_mask == False:
                        self.task_queue_annotation_embedding_generation.put(id)
                    
                    # Check if fusion incluyde this embedding
                    if annotation_info.embedding_set_time > embeddings_time:
                        need_embedding_fusion = True
                
                # ask embedding fusion if needed
                if need_embedding_fusion:
                    self.task_queue_category_embedding_generation.put(time.time())
            
            if not personne_cat_exist:
                self.database_services.RegisterDatabaseCategory("personne")
            
            print("Scan ended")
    
    def _task_loop_update_working_model(self):
        time_since_last_check = 0.0
        while True:
            # sleep
            time.sleep(0.5)

            start = time.time()

            if not self.check_model_update and time_since_last_check < 30.0:
                time_since_last_check += 0.05
                continue

            with self.input_lock:
                if self.input_image is None:
                    time.sleep(5.0)
                    continue

            time_since_last_check = 0.0
            self.check_model_update = False
            # update embeddings
            self._update_working_model()

            print("Working model updated in", time.time() - start, "seconds")

    def _task_loop_inference(self):
        last_input_get_time = time.time()
        last_loop_time = 0.0
        while True:

            # sleep
            if( last_loop_time < self.inference_dt):
                time.sleep(self.inference_dt - last_loop_time)
            start_time = time.time()

            # Getting image to process
            with self.input_lock:

                if(last_input_get_time > self.input_update_time):
                    last_loop_time = time.time() - start_time
                    print("No new image to process")
                    continue

                if self.input_image is None:
                    last_loop_time = time.time() - start_time
                    print("No new image to process")
                    continue

                img_to_process = self.input_image.copy()
                last_input_get_time = time.time()

            # inference
            with self.working_model_lock:
                if self.working_model is None:
                    print("No working model available")
                    time.sleep(10.0)
                    last_loop_time = time.time() - start_time
                    continue
                result = self.working_model(img_to_process)[0]

            # save results
            with self.output_lock:
                self.output_image = img_to_process.copy()
                self.output_result = result
                self.output_update_time = time.time()

            
            # extract result
            extracted_results = list()
            for i in range(len(result.boxes)):
                boxe = result.boxes[i]
                mask = result.masks.data[i].to(dtype=torch.bool)
                bboxes = boxe.xyxy[0]
                class_name = self.category_alias_to_name[self.alias_name_list[int(boxe.cls.item())]]
                confidence = boxe.conf.item() ** self.alias_score_exponent_list[int(boxe.cls.item())]

                if class_name == "__NOTHING__":
                    confidence = confidence ** 3.0

                if(confidence < 0.1 ):
                    continue

                extracted_results.append((bboxes, class_name, confidence, mask))


            # filter nms and nothing
            keep_id = list(range(len(extracted_results)))
            for i in range(len(extracted_results)):
                
                (bboxes, class_name, confidence, mask) = extracted_results[i]
                if class_name == "__NOTHING__":
                    if i in keep_id:
                        keep_id.remove(i)

                for j in range(i+1, len(extracted_results)):

                    (k_bboxes, k_class_name, k_confidence, k_mask) = extracted_results[j]

                    iou_threshold = 0.5

                    if( (i not in keep_id) and (j not in keep_id)):
                        continue

                    if(k_class_name == "__NOTHING__" or class_name == "__NOTHING__"):
                        iou_threshold = 0.8

                    elif(k_class_name == class_name):
                        iou_threshold = 0.4

                    else:
                        continue

                    intersection = torch.count_nonzero(torch.logical_and(mask, k_mask)).item()
                    union = torch.count_nonzero(torch.logical_or(mask, k_mask)).item()
                    iou = float(intersection) / float(union)

                    if( iou > iou_threshold ):
                        if( confidence > k_confidence):
                            if j in keep_id:
                                keep_id.remove(j)
                        else:
                            if i in keep_id:
                                keep_id.remove(i)
            
            keeped_result = [ x for i, x in enumerate(extracted_results) if i in keep_id ]

            # format prediction
            predictions = list()
            for bboxes, class_name, confidence, mask in keeped_result:

                prediction = Prediction()
                prediction_bbox = Bbox()

                prediction.class_name = class_name
                prediction.confidence = confidence

                prediction_bbox.x1 = float(bboxes[0].item())
                prediction_bbox.y1 = float(bboxes[1].item())
                prediction_bbox.x2 = float(bboxes[2].item())
                prediction_bbox.y2 = float(bboxes[3].item())
                prediction.bbox = prediction_bbox

                predictions.append(prediction)

            prediction_result = PredictionResult()
            prediction_result.predictions = predictions
            prediction_result.image = self.cv_bridge.cv2_to_imgmsg(img_to_process, 'bgr8')


            # publish predictions
            self.pub_predictions.publish(prediction_result)

            # get loop time
            last_loop_time = time.time() - start_time

    def _task_loop_detect_new_object(self):
        last_input_get_time = time.time()
        while True:

            # sleep
            time.sleep(5.0)
            print("Task loop for detect new object started")

            # Getting image to process
            with self.output_lock:

                if(last_input_get_time > self.output_update_time):
                    print("No new image to process")
                    continue

                if self.output_image is None:
                    print("No new image to process")
                    continue

                img_to_process = self.output_image.copy()
                yolo_result = self.output_result
                last_input_get_time = time.time()

            # new object detection parameter
            new_object_detection_parameters = NewObjectDetectionParameters()
            new_object_detection_parameters.min_mask_score = 0.7
            new_object_detection_parameters.model_yolo_result_score_exponant = self.alias_score_exponent_list

            with self.sam_model_lock:
                new_object_detection_parameters.model_sam_result = self.sam_model(img_to_process)[0]

            new_object_detection_parameters.model_yolo_result = yolo_result

            # new object detection
            ((unexplained_masks, unexplained_confs, unexplained_estimation_category_label, unexplained_estimation_conf),(img_mask_sam, img_mask_yolo, img_mask_unexplained))= new_object_detection(
                cv_img = img_to_process.copy(),
                parameters = new_object_detection_parameters
            )

            # publish debug images
            img_raw_msg = self.cv_bridge.cv2_to_imgmsg(img_to_process.copy(), 'bgr8')
            img_mask_sam_msg = self.cv_bridge.cv2_to_imgmsg(img_mask_sam, 'bgr8')
            img_mask_yolo_msg = self.cv_bridge.cv2_to_imgmsg(img_mask_yolo, 'bgr8')
            img_mask_unexplained_msg = self.cv_bridge.cv2_to_imgmsg(img_mask_unexplained, 'bgr8')

            self.pub_debug_img_sam.publish(img_mask_sam_msg)
            self.pub_debug_img_yolo.publish(img_mask_yolo_msg)
            self.pub_debug_img_unexplained.publish(img_mask_unexplained_msg)
            self.pub_debug_img_raw.publish(img_raw_msg)

            # generate questions
            if(len(unexplained_masks) > 0):
                image_msg = self.cv_bridge.cv2_to_imgmsg(img_to_process.copy(), 'bgr8')
                image_result = self.database_services.RegisterDatabaseImage(image_msg)

                if( image_result is not None ):

                    image_id = image_result.id
                    if image_id >= 0:

                        for i in range(len(unexplained_masks)):

                            mask = unexplained_masks[i]
                            mask_conf = unexplained_confs[i]
                            estimation_label_id = -1
                            estimation_conf = unexplained_estimation_conf[i]

                            estimation_cluster_label =  unexplained_estimation_category_label[i]

                            if estimation_cluster_label is not None:
                                estimation_label = self.category_alias_to_name[estimation_cluster_label]
                                estimation_label_info = self.database_services.GetDatabaseCategoryByName(estimation_label)

                                if(estimation_label_info is not None):
                                    estimation_label_id = estimation_label_info.info.id
                         
                            visual_prompts = dict(
                                masks=[mask.cpu().numpy().astype(np.float32),],
                                cls=[0,]
                            )
                            with self.embedding_generation_model_lock:
                                embedding = self.embedding_generation_model.generate_vpe(refer_image=img_to_process.copy(), visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, verbose=False)
                            self.database_services.RegisterDatabaseQuestion(image_id, embedding, mask, mask_conf, estimation_label_id, estimation_conf)

def main(args=None):
    rclpy.init()

    node = ExplorationNode()
    print("Node ready")

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()