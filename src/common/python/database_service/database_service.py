from rclpy.node import Node
from interactive_yolo_interfaces.msg import Bbox
from interactive_yolo_interfaces.srv import GetDatabaseCategoryByName, GetDatabaseCategory, GetDatabaseAnnotation, GetDatabaseImage, GetAllDatabaseCategories, SetDatabaseAnnotationEmbedding, SetDatabaseCategoryEmbeddings, SetDatabaseCategoryZeroshotEmbedding, RegisterDatabaseQuestion, RegisterDatabaseAnnotation, RegisterDatabaseCategory, RegisterDatabaseImage, GetDatabaseQuestion, SolveDatabaseQuestion, SetDatabaseAnnotationMask, OpenImage
from tensor_msg_conversion import torchTensorToFloat32Tensor, torchTensorToBoolTensor
import time
import torch
from typing import List
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class DatabaseServices:
    def __init__(self, node:Node):

        self._callback_group = MutuallyExclusiveCallbackGroup()
        
        self.client_GetDatabaseCategory                     = node.create_client(GetDatabaseCategory, 'interactive_yolo/get_database_category', callback_group=self._callback_group)
        self.client_GetDatabaseCategoryByName               = node.create_client(GetDatabaseCategoryByName, 'interactive_yolo/get_database_category_by_name', callback_group=self._callback_group)
        self.client_GetDatabaseAnnotation                   = node.create_client(GetDatabaseAnnotation, 'interactive_yolo/get_database_annotation', callback_group=self._callback_group)
        self.client_GetDatabaseImage                        = node.create_client(GetDatabaseImage, 'interactive_yolo/get_database_image', callback_group=self._callback_group)
        self.client_GetAllDatabaseCategories                = node.create_client(GetAllDatabaseCategories, 'interactive_yolo/get_all_database_categories', callback_group=self._callback_group)
        self.client_SetDatabaseAnnotationEmbedding          = node.create_client(SetDatabaseAnnotationEmbedding, 'interactive_yolo/set_database_annotation_embedding', callback_group=self._callback_group)
        self.client_SetDatabaseCategoryEmbeddings           = node.create_client(SetDatabaseCategoryEmbeddings, 'interactive_yolo/set_database_category_embeddings', callback_group=self._callback_group)
        self.client_SetDatabaseCategoryZeroshotEmbedding    = node.create_client(SetDatabaseCategoryZeroshotEmbedding, 'interactive_yolo/set_database_category_zeroshot_embedding', callback_group=self._callback_group)
        self.client_RegisterDatabaseQuestion                = node.create_client(RegisterDatabaseQuestion, 'interactive_yolo/register_database_question', callback_group=self._callback_group)
        self.client_RegisterDatabaseAnnotation              = node.create_client(RegisterDatabaseAnnotation, 'interactive_yolo/register_database_annotation', callback_group=self._callback_group)
        self.client_RegisterDatabaseCategory                = node.create_client(RegisterDatabaseCategory, 'interactive_yolo/register_database_category', callback_group=self._callback_group)
        self.client_RegisterDatabaseImage                   = node.create_client(RegisterDatabaseImage, 'interactive_yolo/register_database_image', callback_group=self._callback_group)
        self.client_GetDatabaseQuestion                     = node.create_client(GetDatabaseQuestion, 'interactive_yolo/get_database_question', callback_group=self._callback_group)
        self.client_SolveDatabaseQuestion                   = node.create_client(SolveDatabaseQuestion, 'interactive_yolo/solve_database_question', callback_group=self._callback_group)
        self.client_SetDatabaseAnnotationMask               = node.create_client(SetDatabaseAnnotationMask, 'interactive_yolo/set_database_annotation_mask', callback_group=self._callback_group)
        self.client_OpenImage                               = node.create_client(OpenImage, 'interactive_yolo/open_database_image', callback_group=self._callback_group)

    def GetDatabaseCategory(self, id:int)->GetDatabaseCategory.Response:
        
        if not self.client_GetDatabaseCategory.wait_for_service(0.05):
            return None
        
        request = GetDatabaseCategory.Request()
        request.id = id

        future = self.client_GetDatabaseCategory.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def GetDatabaseCategoryByName(self, name:str)->GetDatabaseCategoryByName.Response:
        
        if not self.client_GetDatabaseCategoryByName.wait_for_service(0.05):
            return None
        
        request = GetDatabaseCategoryByName.Request()
        request.name = name

        future = self.client_GetDatabaseCategoryByName.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def GetDatabaseAnnotation(self, id:int)->GetDatabaseAnnotation.Response:
        
        if not self.client_GetDatabaseAnnotation.wait_for_service(0.05):
            return None
        
        request = GetDatabaseAnnotation.Request()
        request.id = id

        future = self.client_GetDatabaseAnnotation.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def GetDatabaseImage(self, id:int)->GetDatabaseImage.Response:
        
        if not self.client_GetDatabaseImage.wait_for_service(0.05):
            return None
        
        request = GetDatabaseImage.Request()
        request.id = id

        future = self.client_GetDatabaseImage.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def GetAllDatabaseCategories(self)->GetAllDatabaseCategories.Response:
        if not self.client_GetAllDatabaseCategories.wait_for_service(0.05):
            return None
        
        request = GetAllDatabaseCategories.Request()

        future = self.client_GetAllDatabaseCategories.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def SetDatabaseAnnotationEmbedding(self, id:int, embedding:torch.Tensor, use_mask:bool)->SetDatabaseAnnotationEmbedding.Response:
        
        if not self.client_SetDatabaseAnnotationEmbedding.wait_for_service(0.05):
            return None
        
        request = SetDatabaseAnnotationEmbedding.Request()
        request.id = id
        request.embedding = torchTensorToFloat32Tensor(embedding)
        request.use_mask = use_mask

        future = self.client_SetDatabaseAnnotationEmbedding.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def SetDatabaseCategoryEmbeddings(self, id:int, embeddings:List[torch.Tensor], embeddings_scores_exponential:List[float] = None)->SetDatabaseCategoryEmbeddings.Response:
        
        if embeddings_scores_exponential is not None:
            embeddings_scores_exponential = [1.0 for x in range(len(embeddings))]

        if not self.client_SetDatabaseCategoryEmbeddings.wait_for_service(0.05):
            return None
        
        request = SetDatabaseCategoryEmbeddings.Request()
        request.id = id
        request.embeddings = [torchTensorToFloat32Tensor(embeddings[i]) for i in range(len(embeddings))]
        request.embeddings_scores_exponential = embeddings_scores_exponential

        future = self.client_SetDatabaseCategoryEmbeddings.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def SetDatabaseCategoryZeroshotEmbedding(self, id:int, embedding:torch.Tensor)->SetDatabaseCategoryZeroshotEmbedding.Response:
        
        if not self.client_SetDatabaseCategoryZeroshotEmbedding.wait_for_service(0.05):
            return None
        
        request = SetDatabaseCategoryZeroshotEmbedding.Request()
        request.id = id
        request.embedding = torchTensorToFloat32Tensor(embedding)

        future = self.client_SetDatabaseCategoryZeroshotEmbedding.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def RegisterDatabaseAnnotation(self, image_id:int, category_id:int, bbox:Bbox)->RegisterDatabaseAnnotation.Response:

        if not self.client_RegisterDatabaseAnnotation.wait_for_service(0.05):
            return None

        request = RegisterDatabaseAnnotation.Request()

        request.category_id = category_id
        request.image_id = image_id
        request.bbox = bbox

        future = self.client_RegisterDatabaseAnnotation.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def RegisterDatabaseCategory(self, name:str )->RegisterDatabaseCategory.Response:

        if not self.client_RegisterDatabaseCategory.wait_for_service(0.05):
            return None

        request = RegisterDatabaseCategory.Request()

        request.name = name

        future = self.client_RegisterDatabaseCategory.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def RegisterDatabaseImage(self, image_msg)->RegisterDatabaseImage.Response:

        if not self.client_RegisterDatabaseImage.wait_for_service(0.05):
            return None

        request = RegisterDatabaseImage.Request()

        request.image = image_msg

        future = self.client_RegisterDatabaseImage.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def GetDatabaseQuestion(self)->GetDatabaseQuestion.Response:

        if not self.client_GetDatabaseQuestion.wait_for_service(0.05):
            return None
        
        request = GetDatabaseQuestion.Request()

        future = self.client_GetDatabaseQuestion.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def RegisterDatabaseQuestion(self, image_id : int, embedding:torch.Tensor, mask:torch.Tensor, mask_confidence:float, 
                                 estimation_category_id:int = -1, estimation_confidence:float = 0.0)->RegisterDatabaseQuestion.Response:

        if not self.client_RegisterDatabaseQuestion.wait_for_service(0.05):
            return None

        request = RegisterDatabaseQuestion.Request()

        request.image_id                = image_id
        request.mask                    = torchTensorToBoolTensor(mask)
        request.mask_confidence         = mask_confidence
        request.estimation_category_id  = estimation_category_id
        request.estimation_confidence   = estimation_confidence
        request.embedding               = torchTensorToFloat32Tensor(embedding)

        future = self.client_RegisterDatabaseQuestion.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def SolveDatabaseQuestion(self, id : int, answer : str, interaction_time:float = 0.0)->SolveDatabaseQuestion.Response:

        if not self.client_SolveDatabaseQuestion.wait_for_service(0.05):
            return None
        
        object_name = answer
        object_valid = True
        if answer is None:
            object_valid = False
            object_name = ""
        
        request = SolveDatabaseQuestion.Request()
        request.id = id
        request.object_valid = object_valid
        request.object_name = object_name
        request.interaction_time = interaction_time

        future = self.client_SolveDatabaseQuestion.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def SetDatabaseAnnotationMask(self, id : int, mask : torch.Tensor ) -> SetDatabaseAnnotationMask.Response:

        if not self.client_SetDatabaseAnnotationMask.wait_for_service(0.05):
            return None
        
        request = SetDatabaseAnnotationMask.Request()
        request.id = id
        request.mask = torchTensorToBoolTensor(mask)

        future = self.client_SetDatabaseAnnotationMask.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()

    def OpenDatabaseImage(self, id : int ) -> OpenImage.Response:

        if not self.client_OpenImage.wait_for_service(0.05):
            return None
        
        request = OpenImage.Request()
        request.id = id

        future = self.client_OpenImage.call_async(request)

        while(not future.done()):
            time.sleep(0.005)
        
        return future.result()