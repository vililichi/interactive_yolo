import torch
import time

class LearnedObject():
    def __init__(self, type:str, embedding:torch.tensor):
        self.type = type
        self.embedding = embedding
        self.creation_time = time.time()

class LearnedCluster():
    def __init__(self, learned_object:LearnedObject):
        self.type = learned_object.type
        self.embedding = learned_object.embedding
        self.nbr_objects = 1

    def same_type(self, learned_object:LearnedObject)->bool:
        return self.type == learned_object.type
    
    def object_distance(self, learned_object:LearnedObject)->float:
        error = torch.dist(self.embedding, learned_object.embedding, p=2).item()

    def add_object(self, learned_object:LearnedObject):
        self.embedding = (self.embedding*self.nbr_objects + learned_object.embedding)/(self.nbr_objects+1)
        self.nbr_objects += 1


