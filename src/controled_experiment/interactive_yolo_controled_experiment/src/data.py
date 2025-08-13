from interactive_yolo_utils import workspace_dir
from .learned_object import LearnedObject
import os
from typing import List
import numpy as np
import cv2
import pickle

class DataManager():

    def __init__(self):

        self.data_dir = os.path.join(workspace_dir(), "controled_experiment", "data")
        self.validation_dir = os.path.join(self.data_dir, "validation")
        self.transcript_dir = os.path.join(self.data_dir, "transcript")
        self.learning_dir = os.path.join(self.data_dir, "learning")

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.isdir(self.validation_dir):
            os.makedirs(self.validation_dir)

        if not os.path.isdir(self.transcript_dir):
            os.makedirs(self.transcript_dir)

        if not os.path.isdir(self.learning_dir):
            os.makedirs(self.learning_dir)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------

    def _get_validation_set_path(self, set_name:str):
        return os.path.join(self.validation_dir, set_name)
    
    def add_validation_set(self, set_name:str):

        path = self._get_validation_set_path(set_name)
        os.makedirs(path)
    
    def list_validation_set(self)->List[str]:
        out = []
        potential_sets = os.listdir(self.validation_dir)
        
        for potential_set in potential_sets:

            path = self._get_validation_set_path(potential_set)
            if os.path.isdir(path):
                out.append(potential_set)
        
        return out
    
    def register_image_in_validation_set(self, set_name:str, image_name:str, cv_image:np.ndarray):
         
        set_path = self._get_validation_set_path(set_name)
        image_path = os.path.join(set_path, image_name+".png")
        cv2.imwrite(image_path, cv_image)

    def list_images(self, set_name:str):
        out = []
        set_path = self._get_validation_set_path(set_name)
        potential_images = os.listdir(set_path)
        
        for potential_image in potential_images:

            if ".png" == potential_image[-4:]:
                out.append(potential_image[:-4])
        
        return out

    def load_image(self, set_name:str, image_name:str )->np.ndarray:
        set_path = self._get_validation_set_path(set_name)
        image_path = os.path.join(set_path, image_name+".png")
        return cv2.imread(image_path)
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------

    def _get_transcript_set_path(self, set_name:str):
        return os.path.join(self.transcript_dir, set_name)
    
    def add_transcript_set(self, set_name:str):

        path = self._get_transcript_set_path(set_name)
        os.makedirs(path)
    
    def list_transcript_set(self)->List[str]:
        out = []
        potential_sets = os.listdir(self.transcript_dir)
        
        for potential_set in potential_sets:

            path = self._get_transcript_set_path(potential_set)
            if os.path.isdir(path):
                out.append(potential_set)
        
        return out
    
    def register_transcript_in_transcript_set(self, set_name:str, transcript_name:str, transcript:str):
         
        set_path = self._get_transcript_set_path(set_name)
        transcript_path = os.path.join(set_path, transcript_name+".txt")
        with open(transcript_path, 'w') as f:
            f.write(transcript)

    def list_transcripts(self, set_name:str):
        out = []
        set_path = self._get_transcript_set_path(set_name)
        potential_transcript = os.listdir(set_path)
        
        for potential_image in potential_transcript:

            if ".txt" == potential_image[-4:]:
                out.append(potential_image[:-4])
        
        return out

    def load_transcript(self, set_name:str, transcript_name:str )->np.ndarray:
        set_path = self._get_transcript_set_path(set_name)
        transcript_path = os.path.join(set_path, transcript_name+".txt")
        with open(transcript_path, 'r') as f:
            return f.read()

    #-----------------------------------------------------------------------------------------------------------------------------------------------------

    def _get_learning_set_path(self, set_name:str):
        return os.path.join(self.learning_dir, set_name)
    
    def add_learning_set(self, set_name:str):
        path = self._get_learning_set_path(set_name)
        os.makedirs(path)

    def list_learning_set(self)->List[str]:
        out = []
        potential_sets = os.listdir(self.learning_dir)
        
        for potential_set in potential_sets:

            path = self._get_learning_set_path(potential_set)
            if os.path.isdir(path):
                out.append(potential_set)
        
        return out
    
    def save_learning(self, set_name:str, learning_name:str, learning:List[LearnedObject]):

        set_path = self._get_learning_set_path(set_name)
        learning_path = os.path.join(set_path, learning_name+".pkl")
        with open(learning_path, 'wb') as file:
            pickle.dump(learning, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_learning(self, set_name:str, learning_name:str) -> List[LearnedObject]:
        set_path = self._get_learning_set_path(set_name)
        learning_path = os.path.join(set_path, learning_name+".pkl")
        with open(learning_path, 'rb') as file:
            return pickle.load(file)

    def list_learnings(self, set_name:str):
        out = []
        set_path = self._get_learning_set_path(set_name)
        potential_learnings = os.listdir(set_path)
        
        for potential_learning in potential_learnings:

            if ".pkl" == potential_learning[-4:]:
                out.append(potential_learning[:-4])
        
        return out