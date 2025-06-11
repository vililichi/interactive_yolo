#from .data_format import *

from typing import Dict, List

import os
import pickle
import cv2
import numpy as np
import time
import json
from threading import Thread, Lock

from typing import Tuple

from rosidl_runtime_py import message_to_ordereddict, set_message_fields
from interactive_yolo_interfaces.msg import DatabaseAnnotationInfo, DatabaseCategoryInfo, DatabaseImageInfo, DatabaseQuestionInfo, Float32Tensor, BoolTensor, Bbox, DatabaseSolvedQuestionInfo
from tensor_msg_conversion import float32TensorToNdarray, boolTensorToNdArray
from interactive_yolo_utils import workspace_dir

def empty_bbox() -> Bbox:
    bbox = Bbox()
    bbox.x1 = 0.0
    bbox.y1 = 0.0
    bbox.x2 = 0.0
    bbox.y2 = 0.0
    return bbox

def empty_tensor()->Float32Tensor:
    tensor = Float32Tensor()
    tensor.shape = []
    tensor.tensor_data = ""
    return tensor

def empty_bool_tensor()->BoolTensor:
    tensor = BoolTensor()
    tensor.shape = []
    tensor.tensor_data = ""
    return tensor

def empty_category_info()->DatabaseCategoryInfo:

    category_info = DatabaseCategoryInfo()

    category_info.id                            = -1
    category_info.name                          = ""
    category_info.images_ids                    = []
    category_info.annotations_ids               = []
    category_info.embeddings                    = []
    category_info.zeroshot_embedding            = empty_tensor()
    category_info.embeddings_set_time           = 0.0
    category_info.zeroshot_embedding_set_time   = 0.0
    category_info.last_member_update_time       = 0.0
    category_info.creation_time                 = 0.0

    return category_info

def empty_image_info()->DatabaseImageInfo:

    image_info = DatabaseImageInfo()

    image_info.id               = -1
    image_info.path             = ""
    image_info.categories_ids   = []
    image_info.annotations_ids  = []
    image_info.last_member_update_time  = 0.0
    image_info.creation_time    = 0.0

    return image_info

def empty_annotation_info()->DatabaseAnnotationInfo:    

    annotation_info = DatabaseAnnotationInfo()

    annotation_info.id                          = -1
    annotation_info.image_id                    = -1
    annotation_info.category_id                 = -1
    annotation_info.bbox                        = empty_bbox()
    annotation_info.embedding                   = empty_tensor()
    annotation_info.mask                        = empty_bool_tensor()
    annotation_info.creation_time               = 0.0
    annotation_info.bbox_set_time               = 0.0
    annotation_info.last_member_update_time     = 0.0
    annotation_info.embedding_set_time          = 0.0
    annotation_info.mask_set_time               = 0.0

    return annotation_info

def empty_question_info()->DatabaseQuestionInfo:    

    question_info = DatabaseQuestionInfo()

    question_info.id                        = -1
    question_info.image_id                  = -1
    question_info.mask                      = empty_bool_tensor()
    question_info.mask_confidence           = 0.0
    question_info.mask_area                 = 0
    question_info.mask_relative_area        = 0.0
    question_info.estimation_category_id    = -1
    question_info.estimation_confidence     = 0.0
    question_info.creation_time             = 0.0
    question_info.embedding                 = empty_tensor()

    return question_info

def empty_solved_question_info()->DatabaseSolvedQuestionInfo:

    solved_question_info = DatabaseSolvedQuestionInfo()

    solved_question_info.category_id           = -1
    solved_question_info.interaction_duration  = 0.0
    solved_question_info.solved_time           = 0.0
    solved_question_info.question              = empty_question_info()

    return solved_question_info

class Database:

    categories : Dict[int, DatabaseCategoryInfo]
    " Dictionnaire contenant les categories, la clé est l'id"
    _last_categories_id : int
    " Dernière id assignée à une categorie"
    _categories_names_to_id : Dict[str,int]
    " Permet de trouver l'id d'une categorie à partie du nom de celle-ci"

    images : Dict[int, DatabaseImageInfo]
    " Dictionnaire contenant les images, la clé est l'id"
    _last_images_id : int
    " Dernière id assignée à une image"

    annotations : Dict[int, DatabaseAnnotationInfo]
    " Dictionnaire contenant les annotations, la clé est l'id"
    _last_annotations_id : int
    " Dernière id assignée à une annotation"

    questions : Dict[int, DatabaseQuestionInfo]
    " Dictionnaire contenant les questions, la clé est l'id"
    solved_questions : Dict[int, DatabaseSolvedQuestionInfo]
    " Dictionnaire contenant les questions résolues, la clé est l'id"
    _last_question_id : int
    " Dernière id assignée à une annotation"

    _data_path:str
    " Dossier contenant les données"

    _images_path:str
    " Dossier contenant les image"

    _json_images_info_path:str
    " Fichier contenant les info sur les images"

    _json_categories_info_path:str
    " Fichier contenant les info sur les catégories"

    _json_annotations_info_path:str
    " Fichier contenant les info sur les annotations"

    _json_questions_info_path:str
    " Fichier contenant les info sur les questions"

    def __init__(self):
        self.categories = dict()
        self._last_categories_id = -1
        self._categories_names_to_id = dict()
        self._categories_lock = Lock()

        self.images = dict()
        self._last_images_id = -1
        self._images_lock = Lock()

        self.annotations = dict()
        self._last_annotations_id = -1
        self._annotations_lock = Lock()

        self.questions = dict()
        self.solved_questions = dict()
        self._last_question_id = -1
        self._questions_lock = Lock()

        self._data_path = os.path.normpath(os.path.join(workspace_dir(),"database"))
        self._images_path = os.path.join(self._data_path, "images")

        self._json_images_info_path         = os.path.join(self._data_path, "images.json")
        self._json_categories_info_path     = os.path.join(self._data_path, "categories.json")
        self._json_annotations_info_path    = os.path.join(self._data_path, "annotations.json")
        self._json_questions_info_path      = os.path.join(self._data_path, "questions.json")

        self._need_save_annotation = False
        self._need_save_category = False
        self._need_save_image = False
        self._need_save_question = False
        self._need_save_lock = Lock()
        self._save_thread = Thread(target=self._save_thread_function, daemon=True)
        self._save_thread.start()

        os.makedirs(self._images_path, exist_ok=True)
        self._load()

    def _save_thread_function(self):
        while True:

            with(self._need_save_lock):
                need_save_annotation = self._need_save_annotation
                need_save_category = self._need_save_category
                need_save_image = self._need_save_image
                need_save_question = self._need_save_question
                self._need_save_annotation = False
                self._need_save_category = False
                self._need_save_image = False
                self._need_save_question = False

            if( need_save_annotation ):
                self._save_annotations_info()

            if( need_save_category ):
                self._save_categories_info()

            if( need_save_image ):
                self._save_images_info()

            if( need_save_question ):
                self._save_questions_info()

            time.sleep(20.0)

    def _generate_image_path(self, id:int)->str:
        filename = str(id).rjust(8,"0")+".png"
        return os.path.join(self._images_path, filename)

    def get_categorie_info(self, id)->DatabaseCategoryInfo:
        with self._categories_lock:
            if not id in self.categories.keys():
                return empty_category_info()
            
            return self.categories[id]

    def get_categorie_info_by_name(self, name:str)->DatabaseCategoryInfo:
        with self._categories_lock:
            if not name in self._categories_names_to_id.keys():
                return empty_category_info()
            
            return self.categories[self._categories_names_to_id[name]]

    def get_image_info(self, id:int)->DatabaseImageInfo:
        with self._images_lock:
            if not id in self.images.keys():
                return empty_image_info()
            
            return self.images[id]

    def get_annotation_info(self, id:int)->DatabaseAnnotationInfo:
        with self._annotations_lock:
            if not id in self.annotations.keys():
                return empty_annotation_info()

            return self.annotations[id]

    def get_all_categories(self)->List[DatabaseCategoryInfo]:
        with self._categories_lock:
            categories = []
            for id in self.categories.keys():
                categories.append(self.categories[id])
            return categories

    def add_categorie(self, name:str )->int:
        with self._categories_lock:
            if name in self._categories_names_to_id.keys() :
                return self._categories_names_to_id[name]
            
            self._last_categories_id += 1
            actual_time = time.time()
            
            category_info = empty_category_info()
            category_info.name                      = name
            category_info.id                        = self._last_categories_id
            category_info.last_member_update_time   = actual_time
            category_info.creation_time             = actual_time

            self.categories[category_info.id] = category_info
            self._categories_names_to_id[name] = category_info.id

            with self._need_save_lock:
                self._need_save_category = True

            return category_info.id

    def add_image(self, image:any )->int:
        with self._images_lock:
            self._last_images_id += 1
            actual_time = time.time()

            image_info = empty_image_info()
            image_info.id                       = self._last_images_id
            image_info.path                     = self._generate_image_path(image_info.id)
            image_info.last_member_update_time  = actual_time
            image_info.creation_time            = actual_time

            cv2.imwrite(image_info.path, image)

            self.images[image_info.id] = image_info

            with self._need_save_lock:
                self._need_save_image = True

            return image_info.id

    def _remove_image(self, image_id )->bool:
        with self._images_lock:
            if( image_id in self.images.keys()):

                image_info = self.images[image_id]
                if os.path.isfile(image_info.path):
                    os.remove(image_info.path)

                self.images.pop(image_id)

                with self._need_save_lock:
                    self._need_save_image = True

                return True
            return False

    def _clean_images(self):
        deprecate_time = time.time() - 604800

        with self._images_lock:
            img_keys = list(self.images.keys())
        for key in img_keys:
            image = self.get_image_info(key)
            if image.id == -1:
                continue
            if len(image.annotations_ids) > 0:
                continue
            if image.last_member_update_time > deprecate_time:
                continue
            if image.creation_time > deprecate_time:
                continue

            question_found = False
            with self._questions_lock:
                for question in self.questions.values():
                    if question.image_id == key:
                        question_found = True
                        break
                if question_found:
                    continue

            self._remove_image(key)

    def add_annotation(self, image_id, category_id, bbox : Bbox )->int:
        with self._annotations_lock:
            if not image_id in self.images.keys():
                return -1
            
            if not category_id in self.categories.keys():
                return -1
            
            self._last_annotations_id += 1
            actual_time = time.time()
            
            annotation_info = empty_annotation_info()

            annotation_info.id = self._last_annotations_id
            annotation_info.bbox = bbox
            annotation_info.image_id = image_id
            annotation_info.category_id = category_id
            annotation_info.bbox_set_time = actual_time
            annotation_info.last_member_update_time = actual_time
            annotation_info.creation_time = actual_time

            image_info = self.get_image_info(image_id)
            category_info = self.get_categorie_info(category_id)

            with self._images_lock:
                image_info.annotations_ids.append(annotation_info.id)
                image_info.last_member_update_time = actual_time
                if( not category_id in image_info.categories_ids):
                    image_info.categories_ids.append(category_id)

            with self._categories_lock:
                category_info.annotations_ids.append(annotation_info.id)
                category_info.last_member_update_time = actual_time
                if( not image_id in category_info.images_ids):
                    category_info.images_ids.append(image_id)

            self.annotations[annotation_info.id] = annotation_info

            with self._need_save_lock:
                self._need_save_category = True
                self._need_save_annotation = True
                self._need_save_image = True

            return annotation_info.id

    def add_question(self, image_id:int,
                     embedding:Float32Tensor,
                     mask:Float32Tensor, mask_confidence:float,
                     estimation_category_id:int=-1, estimation_confidence:float = 0.0)->int:
        
        with self._questions_lock:
            
            self._last_question_id += 1
            actual_time = time.time()

            question_info = empty_question_info()
            question_info.id                = self._last_question_id
            question_info.image_id          = image_id
            question_info.mask              = mask
            question_info.mask_confidence   = mask_confidence
            question_info.creation_time     = actual_time
            question_info.embedding         = embedding

            mask_array = boolTensorToNdArray(question_info.mask)
            total_area = mask_array.shape[0] * mask_array.shape[1]
            mask_area  = int(np.sum(mask_array))

            question_info.mask_area = mask_area
            question_info.mask_relative_area = mask_area/total_area

            question_info.estimation_category_id = estimation_category_id
            question_info.estimation_confidence = estimation_confidence

            self.questions[question_info.id] = question_info
            
            with self._need_save_lock:
                self._need_save_question = True

            return question_info.id

    def add_solved_question(self, question_info:DatabaseImageInfo,
                     category_id:int,
                     interaction_duration:float)->int:
        
        with self._questions_lock:
            
            self._last_question_id += 1
            actual_time = time.time()

            solved_question_info = empty_solved_question_info()
            solved_question_info.question = question_info
            solved_question_info.category_id = category_id
            solved_question_info.interaction_duration = interaction_duration
            solved_question_info.solved_time = actual_time

            self.solved_questions[solved_question_info.question.id] = solved_question_info
            
            with self._need_save_lock:
                self._need_save_question = True

            return question_info.id

    def _compute_question_priority_score(self, question_id:int, actual_time:float = time.time()):

        with self._questions_lock:
            if( question_id in self.questions.keys()):
                question = self.questions[question_id]
                score = 1.0 #question.mask_confidence ** 0.5
                #
                

                emb_score = 1.0 
                for solved_question in self.solved_questions.values():
                    time_offset = ((actual_time - solved_question.solved_time) / 6000000.0)
                    if time_offset >= 1.0:
                        continue

                    question_emb = float32TensorToNdarray(question.embedding)
                    solved_question_emb = float32TensorToNdarray(solved_question.question.embedding)
                    dist = np.linalg.norm(question_emb - solved_question_emb)
                    sub_emb_score = 1.0 - (dist**0.5)
                    sub_emb_score -= time_offset
                    sub_emb_score = max(sub_emb_score, 0.0)
                    emb_score -= sub_emb_score
                    if( emb_score <= 0.0 ):
                        break
    
                confidence_score = (1.0 - (question.estimation_confidence))**0.5
                score *= min(confidence_score, max(emb_score, 0.0))

                score *= question.mask_confidence ** 0.5
                score *= question.mask_relative_area ** 0.25

                score -= ((actual_time - question.creation_time) / 300.0)

                return score
            return 0.0

    def remove_question(self, question_id:int)->bool:
        with self._questions_lock:
            if( question_id in self.questions.keys()):
                self.questions.pop(question_id)

                with self._need_save_lock:
                    self._need_save_question = True

                return True
            return False

    def solve_question(self, question_id:int, object_valid:bool, object_name:str, interaction_time:float)-> Tuple[bool, int, int]:

        new_cat = -1
        new_annotation = -1

        question_info = self.get_question_info(question_id)
        if question_info.id == -1:
            return False, new_cat, new_annotation
        
        category_name = ""
        if not object_valid :
            category_name = "__NOTHING__"
        else:
            category_name = object_name.lower()

        category_id = self.get_categorie_info_by_name(category_name).id
        if(category_id == -1 ):
            category_id = self.add_categorie(category_name)
            new_cat = category_id

        if(category_id == -1 ):
            return False, new_cat, new_annotation
        
        mask = boolTensorToNdArray(question_info.mask)

        bbox = Bbox()
        xs = np.any(mask, axis=1)
        ys = np.any(mask, axis=0)
        y1, y2 = np.where(ys)[0][[0, -1]]
        x1, x2 = np.where(xs)[0][[0, -1]]
        bbox.x1 = float(x1)
        bbox.x2 = float(x2)
        bbox.y1 = float(y1)
        bbox.y2 = float(y2)

        annotation_id = self.add_annotation(image_id=question_info.image_id, category_id=category_id, bbox=bbox)
        new_annotation = annotation_id

        if annotation_id == -1:
            return False, new_cat, new_annotation

        self.set_annotation_mask(annotation_id, question_info.mask)
        self.set_annotation_embedding(annotation_id, question_info.embedding, use_mask=True )

        self.add_solved_question(question_info, new_cat, interaction_time)
        self.remove_question(question_id)

        return True, new_cat, new_annotation

    def get_question_info(self, id:int)->DatabaseQuestionInfo:
        with self._questions_lock:
            if not id in self.questions.keys():
                return empty_question_info()

            return self.questions[id]

    def get_question(self)->DatabaseQuestionInfo:
        best_question_id = -1
        best_question_score = 0.0
        actual_time = time.time()

        with self._questions_lock:
            keys = list(self.questions.keys())

        for key in keys:
            score = self._compute_question_priority_score(key, actual_time)
            if( score < 0.0 ):
                self.remove_question(key)

            if score > best_question_score:
                best_question_score = score
                best_question_id = key

        if best_question_id == -1:
            return empty_question_info()
        else:
            with self._questions_lock:
                return self.questions[best_question_id]

    def _clean_question(self):
        
        with self._questions_lock:
            question_keys = list(self.questions.keys())
        for key in question_keys:
            if self._compute_question_priority_score(key) > 0.0:
                continue

            self.remove_question(key)

    def set_annotation_embedding(self, id:int, embedding:Float32Tensor, use_mask: bool)->bool:
        with self._annotations_lock:
            if not id in self.annotations.keys():
                return False

            self.annotations[id].embedding = embedding
            self.annotations[id].embedding_set_time = time.time()
            self.annotations[id].embedding_use_mask = use_mask

            with self._need_save_lock:
                self._need_save_annotation = True

            return True

    def set_annotation_mask(self, id:int, mask:Float32Tensor)->bool:
        with self._annotations_lock:
            if not id in self.annotations.keys():
                return False

            self.annotations[id].mask = mask
            self.annotations[id].mask_set_time = time.time()

            with self._need_save_lock:
                self._need_save_annotation = True

            return True

    def set_category_embeddings(self, id:int, embeddings:List[Float32Tensor])->bool:   
        with self._categories_lock: 
            if not id in self.categories.keys():
                return False

            self.categories[id].embeddings = embeddings
            self.categories[id].embeddings_set_time = time.time()

            with self._need_save_lock:
                self._need_save_category = True

            return True

    def set_category_zeroshot_embedding(self, id:int, embedding:Float32Tensor)->bool:
        with self._categories_lock:
            if not id in self.categories.keys():
                return False

            self.categories[id].zeroshot_embedding = embedding
            self.categories[id].zeroshot_embedding_set_time = time.time()

            with self._need_save_lock:
                self._need_save_category = True

            return True

    def _save_images_info(self):

        with self._images_lock:
            json_dict = {
                "infos": dict(),
                "info_type": "interactive_yolo_interfaces/DatabaseImageInfo",
                "last_id": self._last_images_id
            }

            for (key, info) in self.images.items():
                json_dict["infos"][key] = message_to_ordereddict(info)

            with open(self._json_images_info_path, "w") as file:
                json.dump(json_dict, file, indent=4)

    def _save_categories_info(self):
        with self._categories_lock:
            json_dict = {
                "infos": dict(),
                "info_type": "interactive_yolo_interfaces/DatabaseCategoryInfo",
                "last_id": self._last_categories_id,
                "categories_names_to_id" :  self._categories_names_to_id
            }

            for (key, info) in self.categories.items():
                json_dict["infos"][key] = message_to_ordereddict(info)

            with open(self._json_categories_info_path, "w") as file:
                json.dump(json_dict, file, indent=4)

    def _save_annotations_info(self):
        with self._annotations_lock:
            json_dict = {
                "infos": dict(),
                "info_type": "interactive_yolo_interfaces/DatabaseAnnotationInfo",
                "last_id": self._last_annotations_id
            }

            for (key, info) in self.annotations.items():
                json_dict["infos"][key] = message_to_ordereddict(info)

            with open(self._json_annotations_info_path, "w") as file:
                json.dump(json_dict, file, indent=4)

    def _save_questions_info(self):
        with self._questions_lock:
            json_dict = {
                "infos": dict(),
                "solved_infos": dict(),
                "info_type": "interactive_yolo_interfaces/DatabaseQuestionInfo",
                "solved_info_type": "interactive_yolo_interfaces/DatabaseSolvedQuestionInfo",
                "last_id": self._last_question_id
            }

            for (key, info) in self.questions.items():
                json_dict["infos"][key] = message_to_ordereddict(info)

            for (key, info) in self.solved_questions.items():
                json_dict["solved_infos"][key] = message_to_ordereddict(info)

            with open(self._json_questions_info_path, "w") as file:
                json.dump(json_dict, file, indent=4)

    def _load(self):
        self._load_annotations_info()
        self._load_categories_info()
        self._load_images_info()
        self._load_questions_info()

        self._clean_question()
        self._clean_images()

    def _load_images_info(self):
        with self._images_lock:
            load_path = self._json_images_info_path
            if( os.path.isfile(load_path) ):
                with open(load_path, "r") as file:
                    json_dict = json.load(file)

                    info_type = json_dict["info_type"]
                    self._last_images_id = int(json_dict["last_id"])
                    self.images = dict()

                    if info_type != "interactive_yolo_interfaces/DatabaseImageInfo":
                        print("Warning: info_type mismatch in images info file. Expected 'interactive_yolo_interfaces/DatabaseImageInfo', got", info_type)
                        return
                    
                    for (key, info) in json_dict["infos"].items():
                        print("load image", key)
                        msg = DatabaseImageInfo()
                        set_message_fields(msg, info)
                        self.images[int(key)] = msg

    def _load_categories_info(self):
        with self._categories_lock:
            load_path = self._json_categories_info_path
            if( os.path.isfile(load_path) ):
                print(load_path)
                with open(load_path, "r") as file:
                    json_dict = json.load(file)

                    info_type = json_dict["info_type"]
                    self._last_categories_id = int(json_dict["last_id"])
                    self._categories_names_to_id = json_dict["categories_names_to_id"]
                    self.categories = dict()

                    if info_type != "interactive_yolo_interfaces/DatabaseCategoryInfo":
                        print("Warning: info_type mismatch in categories info file. Expected 'interactive_yolo_interfaces/DatabaseCategoryInfo', got", info_type)
                        return

                    for (key, info) in json_dict["infos"].items():
                        print("load category", key)
                        msg = DatabaseCategoryInfo()
                        set_message_fields(msg, info)
                        self.categories[int(key)] = msg

    def _load_annotations_info(self):
        with self._annotations_lock:
            load_path = self._json_annotations_info_path
            if( os.path.isfile(load_path) ):
                with open(load_path, "r") as file:
                    json_dict = json.load(file)

                    info_type = json_dict["info_type"]
                    self._last_annotations_id = int(json_dict["last_id"])
                    self.annotations = dict()

                    if info_type != "interactive_yolo_interfaces/DatabaseAnnotationInfo":
                        print("Warning: info_type mismatch in annotations info file. Expected 'interactive_yolo_interfaces/DatabaseAnnotationInfo', got", info_type)
                        return

                    for (key, info) in json_dict["infos"].items():
                        print("load annotation", key)
                        msg = DatabaseAnnotationInfo()
                        set_message_fields(msg, info)
                        self.annotations[int(key)] = msg

    def _load_questions_info(self):
        with self._questions_lock:
            load_path = self._json_questions_info_path
            if( os.path.isfile(load_path) ):
                with open(load_path, "r") as file:
                    json_dict = json.load(file)

                    info_type = json_dict["info_type"]
                    solved_info_type = json_dict["solved_info_type"]
                    self._last_question_id = int(json_dict["last_id"])
                    self.questions = dict()

                    if info_type != "interactive_yolo_interfaces/DatabaseQuestionInfo":
                        print("Warning: info_type mismatch in questions info file. Expected 'interactive_yolo_interfaces/DatabaseQuestionInfo', got", info_type)
                        return
                    if solved_info_type != "interactive_yolo_interfaces/DatabaseSolvedQuestionInfo":
                        print("Warning: solved_info_type mismatch in questions info file. Expected 'interactive_yolo_interfaces/DatabaseSolvedQuestionInfo', got", solved_info_type)
                        return

                    for (key, info) in json_dict["infos"].items():
                        print("load question", key)
                        msg = DatabaseQuestionInfo()
                        set_message_fields(msg, info)
                        self.questions[int(key)] = msg

                    for (key, info) in json_dict["solved_infos"].items():
                        print("load solved question", key)
                        msg = DatabaseSolvedQuestionInfo()
                        set_message_fields(msg, info)
                        self.solved_questions[int(key)] = msg
