from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor
import torch
import numpy as np
from typing import List, Dict

from .model_loader import yoloe_model, sam_model
from .learned_object import LearnedCluster, LearnedObject
from .question import Question
from .new_object_detection_helper import NewObjectDetectionParameters, new_object_detection

class Model():

    def __init__(self, classes_list=List[str]):
        self.yoloe_model = yoloe_model()
        self.sam_model = sam_model()

        self.classes_list:List[str] = classes_list
        self.classes_clusters:Dict[str, List[LearnedCluster]] = {}
        self.cluster_dist_threshold = 0.5

        for class_name in self.classes_list:
            class_embedding = self.generate_embeddings_from_class_name(class_name)
            self.add_learned_object(LearnedObject(class_name, class_embedding))

    def generate_embeddings_from_class_name(self, class_name:str)->torch.tensor:
        return self.yoloe_model.get_text_pe([class_name,]).cpu()
    
    def generate_embeddings_from_mask(self, mask:np.ndarray, refer_image)->torch.tensor:

        visual_prompts = dict(
            masks=[mask.astype(np.float32),],
            cls=[0,]
        )

        return self.yoloe_model.generate_vpe(refer_image=refer_image, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, verbose=False).cpu()

    def add_learned_object(self, learned_object:LearnedObject):

        if learned_object.type not in self.classes_list:
            return

        if learned_object.type in self.classes_clusters:
            best_id = -1
            best_distance = self.cluster_dist_threshold
            for i in range(len(self.classes_clusters[learned_object.type])):
                distance = self.classes_clusters[learned_object.type][i].object_distance(learned_object)
                if distance < best_distance:
                    best_id = i
                    best_distance = distance
            
            if best_id == -1:
                self.classes_clusters[learned_object.type].append(LearnedCluster(learned_object))
            else:
                self.classes_clusters[learned_object.type][best_id].add_object(learned_object)
        else:
            self.classes_clusters[learned_object.type] = [LearnedCluster(learned_object),]

    def configure_model_for_detection(self):

        pe_list = []
        self.alias_name_list = []
        self.category_alias_to_name = dict()

        for class_name in self.classes_list:
            
            for itt in range(len(self.classes_clusters[class_name])):

                alias_name = "__CLUSTER" + str(itt) + "__" + class_name

                self.alias_name_list.append(alias_name)
                pe_list.append(self.classes_clusters[class_name][itt].embedding)

                self.category_alias_to_name[alias_name] = class_name
        
        self.pe = torch.cat(pe_list, dim=1)
        self.yoloe_model.set_classes(self.alias_name_list, self.pe)

    def generate_question(self, image:np.ndarray)->List[Question]:

        questions = []

        parameters = NewObjectDetectionParameters()
        parameters.model_yolo_result = self.yoloe_model(image)[0]
        parameters.model_sam_result = self.sam_model(image)[0]

        (unexplained_masks, unexplained_confs, unexplained_estimation_category_label, unexplained_estimation_conf) = new_object_detection(parameters)
        
        for i in range(len(unexplained_masks)):

            mask = unexplained_masks[i].cpu().numpy().astype(np.float32)
            mask_conf = unexplained_confs[i]
            estimation_conf = unexplained_estimation_conf[i]
            embedding = self.generate_embeddings_from_mask(mask, image)

            questions.append(Question(mask = mask, embedding = embedding, mask_conf = mask_conf, explain_score = estimation_conf, image_shape=image.shape))
        
        return questions

