from cv_bridge import CvBridge

import rclpy
import rclpy.executors
from rclpy.node import Node
from interactive_yolo_interfaces.srv import GetDatabaseAnnotation, GetDatabaseCategory, GetDatabaseCategoryByName, GetDatabaseImage, GetDatabaseQuestion, GetAllDatabaseCategories
from interactive_yolo_interfaces.srv import RegisterDatabaseAnnotation, RegisterDatabaseCategory, RegisterDatabaseImage, RegisterDatabaseQuestion
from interactive_yolo_interfaces.srv import SetDatabaseAnnotationEmbedding, SetDatabaseAnnotationMask, SetDatabaseCategoryEmbeddings, SetDatabaseCategoryZeroshotEmbedding, SolveDatabaseQuestion
from interactive_yolo_interfaces.msg import DatabaseUpdateNotifaction

from rclpy.callback_groups import ReentrantCallbackGroup

from .src.database import Database

class AnnotationDatabaseNode(Node):
    def __init__(self):
        super().__init__('annotation_database_node')

        self.database = Database()
        self.cv_bridge = CvBridge()

        self._callback_group = ReentrantCallbackGroup()

        self.pub_annotation_update_notification     = self.create_publisher(DatabaseUpdateNotifaction, 'interactive_yolo/database_annotation_update_notification'   , 10, callback_group=self._callback_group)
        self.pub_image_update_notification          = self.create_publisher(DatabaseUpdateNotifaction, 'interactive_yolo/database_image_update_notification'        , 10, callback_group=self._callback_group)
        self.pub_category_update_notification       = self.create_publisher(DatabaseUpdateNotifaction, 'interactive_yolo/database_category_update_notification'     , 10, callback_group=self._callback_group)

        self.srv_GetDatabaseAnnotation      = self.create_service(GetDatabaseAnnotation        , 'interactive_yolo/get_database_annotation'          , self.callback_GetDatabaseAnnotation       , callback_group=self._callback_group)
        self.srv_GetDatabaseCategory        = self.create_service(GetDatabaseCategory          , 'interactive_yolo/get_database_category'            , self.callback_GetDatabaseCategory         , callback_group=self._callback_group)
        self.srv_GetDatabaseCategoryByName  = self.create_service(GetDatabaseCategoryByName    , 'interactive_yolo/get_database_category_by_name'    , self.callback_GetDatabaseCategoryByName   , callback_group=self._callback_group)
        self.srv_GetDatabaseImage           = self.create_service(GetDatabaseImage             , 'interactive_yolo/get_database_image'               , self.callback_GetDatabaseImage            , callback_group=self._callback_group)
        self.srv_GetAllDatabaseCategories   = self.create_service(GetAllDatabaseCategories     , 'interactive_yolo/get_all_database_categories'      , self.callback_GetAllDatabaseCategories    , callback_group=self._callback_group)
        self.srv_GetDatabaseQuestion        = self.create_service(GetDatabaseQuestion          , 'interactive_yolo/get_database_question'            , self.callback_GetDatabaseQuestion         , callback_group=self._callback_group)

        self.srv_RegisterDatabaseAnnotation = self.create_service(RegisterDatabaseAnnotation   , 'interactive_yolo/register_database_annotation'     , self.callback_RegisterDatabaseAnnotation  , callback_group=self._callback_group)
        self.srv_RegisterDatabaseCategory   = self.create_service(RegisterDatabaseCategory     , 'interactive_yolo/register_database_category'       , self.callback_RegisterDatabaseCategory    , callback_group=self._callback_group)
        self.srv_RegisterDatabaseImage      = self.create_service(RegisterDatabaseImage        , 'interactive_yolo/register_database_image'          , self.callback_RegisterDatabaseImage       , callback_group=self._callback_group)
        self.srv_RegisterDatabaseQuestion   = self.create_service(RegisterDatabaseQuestion     , 'interactive_yolo/register_database_question'       , self.callback_RegisterDatabaseQuestion    , callback_group=self._callback_group)

        self.srv_SetDatabaseAnnotationEmbedding         = self.create_service(SetDatabaseAnnotationEmbedding        , 'interactive_yolo/set_database_annotation_embedding'          , self.callback_SetDatabaseAnnotationEmbedding      , callback_group=self._callback_group)
        self.srv_SetDatabaseAnnotationMask              = self.create_service(SetDatabaseAnnotationMask             , 'interactive_yolo/set_database_annotation_mask'               , self.callback_SetDatabaseAnnotationMask           , callback_group=self._callback_group)
        self.srv_SetDatabaseCategoryEmbeddings          = self.create_service(SetDatabaseCategoryEmbeddings         , 'interactive_yolo/set_database_category_embeddings'           , self.callback_SetDatabaseCategoryEmbeddings       , callback_group=self._callback_group)
        self.srv_SetDatabaseCategoryZeroshotEmbedding   = self.create_service(SetDatabaseCategoryZeroshotEmbedding  , 'interactive_yolo/set_database_category_zeroshot_embedding'   , self.callback_SetDatabaseCategoryZeroshotEmbedding, callback_group=self._callback_group)

        self.srv_SolveDatabaseQuestion = self.create_service(SolveDatabaseQuestion, 'interactive_yolo/solve_database_question', self.callback_SolveDatabaseQuestion , callback_group=self._callback_group)

    def callback_GetDatabaseAnnotation(self, request: GetDatabaseAnnotation.Request, response: GetDatabaseAnnotation.Response):

        print( "get_annotation_info[",request.id,"]" )
        response.info = self.database.get_annotation_info(request.id)

        print("done")
        return response

    def callback_GetDatabaseCategory(self, request: GetDatabaseCategory.Request, response: GetDatabaseCategory.Response):
        
        print( "get_categorie_info[",request.id,"]" )
        response.info = self.database.get_categorie_info(request.id)

        print("done")
        return response

    def callback_GetDatabaseCategoryByName(self, request: GetDatabaseCategoryByName.Request, response: GetDatabaseCategoryByName.Response):

        print( "get_categorie_info_by_name[",request.name,"]" )
        response.info = self.database.get_categorie_info_by_name(request.name)

        print("done")
        return response

    def callback_GetDatabaseImage(self, request: GetDatabaseImage.Request, response: GetDatabaseImage.Response):

        print( "get_image_info[",request.id,"]" )
        response.info = self.database.get_image_info(request.id)

        print("done")
        return response

    def callback_GetDatabaseQuestion(self, request: GetDatabaseQuestion.Request, response: GetDatabaseQuestion.Response):

        print( "get_question_info[]" )
        response.info, response.score = self.database.get_question()

        print("done")
        return response

    def callback_GetAllDatabaseCategories(self, request: GetAllDatabaseCategories.Request, response: GetAllDatabaseCategories.Response):
        print( "get_all_categories" )
        response.infos = self.database.get_all_categories()

        print("done")
        return response

    def callback_RegisterDatabaseAnnotation(self, request: RegisterDatabaseAnnotation.Request, response: RegisterDatabaseAnnotation.Response):

        print( "register_annotation[",request.image_id,", ",request.category_id,", bbox]" )
        response.id = self.database.add_annotation(request.image_id, request.category_id, request.bbox)

        if( response.id != -1):
            print("send notification")
            notification_annotation = DatabaseUpdateNotifaction()
            notification_annotation.id = response.id
            notification_annotation.type = DatabaseUpdateNotifaction.REGISTERED
            self.pub_annotation_update_notification.publish(notification_annotation)

            notification_image = DatabaseUpdateNotifaction()
            notification_image.id = request.image_id
            notification_image.type = DatabaseUpdateNotifaction.MEMBER_UPDATE
            self.pub_image_update_notification.publish(notification_image)

            notification_category = DatabaseUpdateNotifaction()
            notification_category.id = request.category_id
            notification_category.type = DatabaseUpdateNotifaction.MEMBER_UPDATE
            self.pub_category_update_notification.publish(notification_category)

        print("done")
        return response

    def callback_RegisterDatabaseCategory(self, request: RegisterDatabaseCategory.Request, response: RegisterDatabaseCategory.Response):

        print( "register_category[",request.name,"]" )
        response.id = self.database.add_categorie(request.name)

        print("send notification")
        notification = DatabaseUpdateNotifaction()
        notification.id = response.id
        notification.type = DatabaseUpdateNotifaction.REGISTERED
        self.pub_category_update_notification.publish(notification)

        print("done")
        return response

    def callback_RegisterDatabaseImage(self, request: RegisterDatabaseImage.Request, response: RegisterDatabaseImage.Response):
        
        print( "register_image[image]" )
        cv_image = self.cv_bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        response.id = self.database.add_image(cv_image)

        print("send notification")
        notification = DatabaseUpdateNotifaction()
        notification.id = response.id
        notification.type = DatabaseUpdateNotifaction.REGISTERED
        self.pub_image_update_notification.publish(notification)

        print("done")
        return response

    def callback_RegisterDatabaseQuestion(self, request: RegisterDatabaseQuestion.Request, response: RegisterDatabaseQuestion.Response):
        
        print( "register_question[question]" )
        response.id = self.database.add_question(request.image_id, request.embedding, request.mask, request.mask_confidence, request.estimation_category_id, request.estimation_confidence)

        print("done")
        return response

    def callback_SetDatabaseAnnotationEmbedding(self, request: SetDatabaseAnnotationEmbedding.Request, response: SetDatabaseAnnotationEmbedding.Response):
        
        print( "set_annotation_embedding[",request.id,"]" )
        response.success = self.database.set_annotation_embedding(request.id, request.embedding, request.use_mask)

        if( response.success):
            print("send notification")
            notification = DatabaseUpdateNotifaction()
            notification.id = request.id
            notification.type = DatabaseUpdateNotifaction.EMBEDDING_UPDATE
            self.pub_annotation_update_notification.publish(notification)

        print("done")
        return response

    def callback_SetDatabaseAnnotationMask(self, request: SetDatabaseAnnotationMask.Request, response: SetDatabaseAnnotationMask.Response):
        
        print( "set_annotation_embedding[",request.id,"]" )
        response.success = self.database.set_annotation_mask(request.id, request.mask)

        if( response.success):
            print("send notification")
            notification = DatabaseUpdateNotifaction()
            notification.id = request.id
            notification.type = DatabaseUpdateNotifaction.MEMBER_UPDATE
            self.pub_annotation_update_notification.publish(notification)

        print("done")
        return response

    def callback_SetDatabaseCategoryEmbeddings(self, request: SetDatabaseCategoryEmbeddings.Request, response: SetDatabaseCategoryEmbeddings.Response):
        
        print( "set_category_embeddings[",request.id,"]" )
        response.success = self.database.set_category_embeddings(request.id, request.embeddings, request.embeddings_scores_exponential)

        if( response.success):
            print("send notification")
            notification = DatabaseUpdateNotifaction()
            notification.id = request.id
            notification.type = DatabaseUpdateNotifaction.EMBEDDING_UPDATE
            self.pub_category_update_notification.publish(notification)

        print("done")
        return response

    def callback_SetDatabaseCategoryZeroshotEmbedding(self, request: SetDatabaseCategoryZeroshotEmbedding.Request, response: SetDatabaseCategoryZeroshotEmbedding.Response):
        print( "set_category_zeroshot_embedding[",request.id,"]" )
        response.success = self.database.set_category_zeroshot_embedding(request.id, request.embedding)

        if( response.success):
            print("send notification")
            notification = DatabaseUpdateNotifaction()
            notification.id = request.id
            notification.type = DatabaseUpdateNotifaction.EMBEDDING_UPDATE
            self.pub_category_update_notification.publish(notification)

        print("done")
        return response

    def callback_SolveDatabaseQuestion(self, request: SolveDatabaseQuestion.Request, response: SolveDatabaseQuestion.Response):
        print( "solve_question[",request.id,',',request.object_name,"]" )
        response.success, new_cat, new_annotation = self.database.solve_question(request.id, request.object_valid, request.object_name, request.interaction_time)

        if( new_cat != -1 ):
            print("send notification")
            notification = DatabaseUpdateNotifaction()
            notification.id = new_cat
            notification.type = DatabaseUpdateNotifaction.REGISTERED
            self.pub_category_update_notification.publish(notification)

        if( new_annotation != -1):
            annotation_info = self.database.get_annotation_info(new_annotation)
            print("send notification")
            notification_annotation = DatabaseUpdateNotifaction()
            notification_annotation.id = annotation_info.id
            notification_annotation.type = DatabaseUpdateNotifaction.REGISTERED
            self.pub_annotation_update_notification.publish(notification_annotation)

            notification_image = DatabaseUpdateNotifaction()
            notification_image.id = annotation_info.image_id
            notification_image.type = DatabaseUpdateNotifaction.MEMBER_UPDATE
            self.pub_image_update_notification.publish(notification_image)

            notification_category = DatabaseUpdateNotifaction()
            notification_category.id = annotation_info.category_id
            notification_category.type = DatabaseUpdateNotifaction.MEMBER_UPDATE
            self.pub_category_update_notification.publish(notification_category)

            notification = DatabaseUpdateNotifaction()
            notification.id = request.id
            notification.type = DatabaseUpdateNotifaction.EMBEDDING_UPDATE
            self.pub_annotation_update_notification.publish(notification)

        

        print("done")
        return response

    def close(self):
        self.database.close()

def main(args=None):
    
    try:
        rclpy.init()
        executor = rclpy.executors.MultiThreadedExecutor(8)

        node = AnnotationDatabaseNode()
        executor.add_node(node)
        print("Node ready")

        executor.spin()
        rclpy.shutdown()

    finally:
        node.close()

if __name__ == '__main__':
    main()