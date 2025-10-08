import torch
import clip
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple
import math

from .model_loader import fast_sam_model
from .learned_object import LearnedCluster, LearnedObject
from .question import Question
from ultralytics.utils.ops import scale_image

import time

class Model():

    def __init__(self, classes_list=List[str]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_device = self.device
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.clip_device, jit=True)

        self.sam_model = fast_sam_model()
        #self.sam_model.eval()
        self.zero_shot_objects_cache = None

        self.classes_list:List[str] = classes_list

        self.embedding_label_cache_update = False
        self.embedding_label_cache_blacklist = []

        self.classes_clusters:Dict[str, List[LearnedCluster]] = {}
        self.classes_bbox_score_normal_curve:Dict[Tuple[float,float,List]] = {}
        self.reset_learning()

    def generate_embeddings_from_class_name(self, class_name:str)->torch.tensor:
        with torch.no_grad():
            text = clip.tokenize([class_name,]).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            out = text_features[[0,]].cpu()
            del text_features

            return out
    
    def generate_embeddings_from_class_names(self, class_names:List[str])->torch.tensor:
        with torch.no_grad():
            texts = clip.tokenize(class_names).to(self.clip_device)
            texts_features = self.clip_model.encode_text(texts)
            texts_features = texts_features / texts_features.norm(dim=1, keepdim=True)
            out = [texts_features[[i,]].cpu() for i in range(texts_features.size()[0])]
            del texts
            del texts_features

            return out
    
    def get_bbox(self, mask)->tuple:
        ys = np.any(mask, axis=1)
        xs = np.any(mask, axis=0)
        y1, y2 = np.where(ys)[0][[0, -1]]
        x1, x2 = np.where(xs)[0][[0, -1]]

        return (x1, y1, x2, y2)
    
    def get_iou(self, bbox1, bbox2):
        (x1_1, y1_1, x1_2, y1_2) = bbox1
        (x2_1, y2_1, x2_2, y2_2) = bbox2
        x_min = max(x1_1, x2_1)
        x_max = min(x1_2, x2_2)
        y_min = max(y1_1, y2_1)
        y_max = min(y1_2, y2_2)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max-x_min)*(y_max-y_min)
        area_1 = (x1_2-x1_1)*(y1_2-y1_1)
        area_2 = (x2_2-x2_1)*(y2_2-y2_1)
        union = area_1 + area_2 - intersection

        if union == 0.0:
            return 0.0
        
        return intersection / union
    
    def generate_image_from_bbox(self, bbox:Tuple[int,int,int,int], ref_image:np.ndarray, margin = 10)->np.ndarray:
        cv_image = ref_image.copy()

        (x1, y1, x2, y2) =  bbox

        y1 = max(y1-margin, 0)
        y2 = min(y2+margin, cv_image.shape[0]-1)
        x1 = max(x1-margin, 0)
        x2 = min(x2+margin, cv_image.shape[1]-1)

        while (y2 - y1) < 2:
            y1 = max(y1-1, 0)
            y2 = min(y2+1, cv_image.shape[0]-1)

        while (x2 - x1) < 2:
            x1 = max(x1-1, 0)
            x2 = min(x2+1, cv_image.shape[1]-1)

        cv_image = cv_image[y1:y2, x1:x2, :]

        return cv_image
    
    def generate_mask_from_bbox(self, bbox:Tuple[int,int,int,int], ref_image:np.ndarray):
        out = np.zeros(ref_image.shape[:2], dtype=np.bool_)
        (x1, y1, x2, y2) =  bbox
        out[y1:y2, x1:x2] = True
        return out
        
    def masks_to_bboxes(self,segments:dict, iou_threshold = 0.8)->Tuple[List[Tuple[int,int,int,int]], List[float]]:

        segments.sort(key=lambda item: item['score'], reverse = True)

        kept_bbox = []
        out = []
        for segment in segments:
            bbox = self.get_bbox(segment['mask'])
            keep = True
            for bbox_2 in kept_bbox:
                if self.get_iou(bbox, bbox_2) > iou_threshold:
                    keep = False
                    break
            if keep:
                kept_bbox.append(bbox)
                out.append({'bbox':bbox, 'score':segment['score']})
        
        return out
 
    def cv_to_pil(self, cv_image:np.ndarray)->Image:
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def generate_embeddings_from_bbox(self, bbox:Tuple[int,int,int,int], refer_image)->torch.tensor:
        mask_image = self.cv_to_pil(self.generate_image_from_bbox(bbox, refer_image, 0))

        with torch.no_grad():
            image = self.clip_preprocess(mask_image).unsqueeze(0).to(self.clip_device )
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            image_features_cpu = image_features.cpu()
            del image
            del image_features

            return image_features_cpu

    def generate_embeddings_from_bboxes(self, bboxes:List[Tuple[int,int,int,int]], refer_image)->torch.tensor:
        return [ self.generate_embeddings_from_bbox(bbox, refer_image) for bbox in bboxes ]

    def reset_learning(self):
        self.ready_for_detection = False
        self.classes_clusters = {}
        self.classes_bbox_score_normal_curve = {}

        self.reset_embedding_label_cache()

        if self.zero_shot_objects_cache is None:
            self.zero_shot_objects_cache = []
            class_embeddings = self.generate_embeddings_from_class_names(self.classes_list)
            for i in range(len(self.classes_list)):
                class_embedding = class_embeddings[i]
                class_name = self.classes_list[i]
                self.zero_shot_objects_cache.append(LearnedObject(class_name, class_embedding))
            
        for learned_object in self.zero_shot_objects_cache:
            self.add_learned_object(learned_object)

    def add_learned_object(self, learned_object:LearnedObject, default_1_in_bbox_score:int = 3):

        if learned_object.type not in self.classes_list:
            return
        
        self.ready_for_detection = False

        if learned_object.type not in self.embedding_label_cache_blacklist:
            self.reset_embedding_label_cache()

        if learned_object.type in self.classes_clusters:
            self.classes_clusters[learned_object.type].append(LearnedCluster(learned_object))
        else:
            self.classes_clusters[learned_object.type] = [LearnedCluster(learned_object),]

        if learned_object.type in self.classes_bbox_score_normal_curve.keys():
            mean, score_list = self.classes_bbox_score_normal_curve[learned_object.type]
            score_list.append(learned_object.seg_score)
            np_score_list = np.asarray(score_list, dtype=np.float32)
            mean = np.mean(np_score_list)
            self.classes_bbox_score_normal_curve[learned_object.type] = (mean, score_list)
        else:

            score_list = [learned_object.seg_score,]
            for _ in range(default_1_in_bbox_score):
                score_list.append(1.0)
            mean = learned_object.seg_score
            self.classes_bbox_score_normal_curve[learned_object.type] = (mean, score_list)
        
    def get_bbox_score_fit(self, class_name:str, score:float)->float:
        
        mean = 1.0
        if class_name in self.classes_bbox_score_normal_curve.keys():
            mean, _ = self.classes_bbox_score_normal_curve[class_name]
        
        z = score-mean

        #return 1.0-abs(z)

        prob = 0.5+0.5*math.sqrt(1-math.exp(-0.6366197723676*z*z))
        return 2.0*(1.0-prob)

    def reset_embedding_label_cache(self):
        self.embedding_label_cache_update = True

    def generate_embedding_label_cache(self, class_blacklist = [])->Tuple[List[str], torch.Tensor]:

        if class_blacklist != self.embedding_label_cache_blacklist or self.embedding_label_cache_update:

            self.embedding_label_cache_classes = []
            self.embedding_label_cache_classes_info = []
            embeddings = []

            i = 0
            for class_name in self.classes_list:

                if class_name in class_blacklist:
                    continue

                start_i = i
                
                for itt in range(len(self.classes_clusters[class_name])):

                    self.embedding_label_cache_classes.append(class_name)
                    embeddings.append(self.classes_clusters[class_name][itt].embedding)
                    i += 1
                
                end_i = i
                self.embedding_label_cache_classes_info.append(
                    {
                        "name":class_name,
                        "start":start_i,
                        "end":end_i
                    }
                )

            with torch.no_grad():
                self.embedding_label_cache_tensor = torch.cat(embeddings, dim=0)

            self.embedding_label_cache_update = False
            self.embedding_label_cache_blacklist = class_blacklist
        
        return self.embedding_label_cache_classes, self.embedding_label_cache_tensor, self.embedding_label_cache_classes_info

    def get_embedding_label_fast(self, embedding:torch.Tensor, k=1, class_blacklist = [])->Tuple[str,float]:
        with torch.no_grad():

            classes, expended_embeddings, _ = self.generate_embedding_label_cache(class_blacklist)
            real_k = min(k, len(classes))

            expended_embedding = embedding.expand(len(classes),-1)
            similarities = torch.nn.functional.relu(torch.nn.functional.cosine_similarity(expended_embedding, expended_embeddings))
            values, indices = torch.topk(similarities, k=real_k, largest=True)

            final_result = {}
            for i in range(real_k):
                similarity = values[i].item()
                classe = classes[indices[i].item()]
                if classe not in final_result.keys():
                    final_result[classe] = similarity
                else:
                    final_result[classe] += similarity
            
            max_label = 'None'
            max_score = 0.0
            for label in final_result.keys():
                score = final_result[label]/min(real_k, 2*len(self.classes_clusters[label]))
                if score > max_score:
                    max_label = label
                    max_score = score

            return(max_label, max_score)
        
    def get_embedding_label_complete(self, embedding:torch.Tensor, k=1, class_blacklist = [], use_softmax:bool = True)->Tuple[str,float]:
        with torch.no_grad():

            classes, expended_embeddings, classes_info = self.generate_embedding_label_cache(class_blacklist)
            real_k = min(k, len(classes))

            expended_embedding = embedding.expand(len(classes),-1)
            similarities = torch.nn.functional.cosine_similarity(expended_embedding, expended_embeddings)

            local_class_list = []
            local_class_score = []
            for class_info in classes_info:

                class_similarities = similarities[class_info['start']:class_info['end']]

                real_k = min(k, class_similarities.size()[0])
                values, _ = torch.topk(class_similarities, k=real_k, largest=True)
                value = torch.mean(values).item()

                local_class_list.append(class_info['name'])
                local_class_score.append(value)

            if use_softmax:
                softmax_score = torch.nn.functional.softmax(torch.FloatTensor(local_class_score), dim=0)
                for i in range(len(local_class_score)):
                    local_class_score[i] = max(min(softmax_score[i].item(), local_class_score[i]),0.0)
            else:
                for i in range(len(local_class_score)):
                    local_class_score[i] = max(local_class_score[i], 0.0)

            return(local_class_score, local_class_list)

    def generate_bboxes(self, cv_image):

        results_sam = self.sam_model(cv_image)[0]
        packed_masks = self.extract_resized_masks_np(results_sam)
        masks_sam = [ packed_masks[i].astype(np.bool_) for i in range(packed_masks.shape[0])]
        confs_sam = [ x.item() for x in torch.unbind(results_sam.boxes.conf.cpu())]
        del results_sam

        masks = []
        for i in range(len(masks_sam)):
            mask = masks_sam[i]
            score = confs_sam[i]
            masks.append({'mask':mask, 'score':score})

        return self.masks_to_bboxes(masks)

    def generate_question(self, image:np.ndarray)->List[Question]:

        t_1 = time.time()
        questions = []

        t_2 = time.time()
        sam_result = self.generate_bboxes(image)
        bboxes_sam = [result['bbox'] for result in sam_result]
        confs_sam = [result['score'] for result in sam_result]

        t_3 = time.time()
        embedding_sam = self.generate_embeddings_from_bboxes(bboxes_sam, image)

        t_4 = time.time()
        label_sam = []
        score_sam = []
        for embedding  in embedding_sam:
            label, score = self.get_embedding_label_fast(embedding, k=1)
            label_sam.append(label)
            score_sam.append(score)
        
        t_5 = time.time()
        for i in range(len(bboxes_sam)):

            mask = self.generate_mask_from_bbox(bboxes_sam[i], image)
            bbox = bboxes_sam[i]
            mask_conf = confs_sam[i]
            estimation_conf = score_sam[i]
            embedding = embedding_sam[i]

            questions.append(Question(mask = mask, embedding = embedding, mask_conf = mask_conf, explain_score = estimation_conf, image_shape=image.shape, bbox=bbox))

        t_6 = time.time()
        #print("total_time = ",t_6-t_1," s")
        #print("config_time = ",t_2-t_1," s")
        #print("sam_time = ",t_3-t_2," s")
        #print("embedding_time = ",t_4-t_3," s")
        #print("classification_time = ",t_5-t_4," s")
        #print("question_time = ",t_6-t_5," s")
        
        return questions
    
    def extract_resized_masks_np(self, result):
    
        if result.masks is None:
            return None
        
        masks = result.masks.data.cpu().detach().numpy()
        masks = np.moveaxis(masks, 0, -1)
        masks = scale_image(masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0)

        return masks
        
    def extract_resized_masks_torch(self, result):

        masks = torch.from_numpy(self.extract_resized_masks_np(result)).to(device=self.device, dtype=torch.bool)

        return masks

    def predict_embeddings_generation(self, image):
        sam_result = self.generate_bboxes(image)
        bboxes_sam = [result['bbox'] for result in sam_result]
        bboxes_score = [result['score'] for result in sam_result]
        embedding_sam = self.generate_embeddings_from_bboxes(bboxes_sam, image)
        return embedding_sam, bboxes_sam, bboxes_score
    
    def predict_embeddings_classification(self, sam_result, class_to_ignore: List[str] = [], min_score:float = 0.001, exclusive:bool=False, bbox_score_coef = 0.3):
        embedding_sam, bboxes_sam, bboxes_score = sam_result
        label_result = []
        score_result = []
        bbox_result = []

        for i  in range(len(bboxes_sam)):

            local_class_score, local_class_list = self.get_embedding_label_complete(embedding_sam[i], class_blacklist=class_to_ignore, k=5, use_softmax=exclusive)

            for j in range(len(local_class_score)):
                total_score = (self.get_bbox_score_fit(local_class_list[j], bboxes_score[i]) ** bbox_score_coef ) * local_class_score[j]
                if total_score < min_score:
                    continue
                label_result.append(local_class_list[j])
                score_result.append(total_score)
                bbox_result.append(bboxes_sam[i])

        return [{"bbox":bbox_result[i], "label":label_result[i], "score":score_result[i]} for i in range(len(bbox_result))]

    def predict(self, image, class_to_ignore: List[str] = [], min_score:float = 0.001, exclusive:bool=False, bbox_score_coef = 0.3):

        sam_result = self.predict_embeddings_generation(image)
        return self.predict_embeddings_classification(sam_result, class_to_ignore, min_score, exclusive=exclusive, bbox_score_coef=bbox_score_coef)