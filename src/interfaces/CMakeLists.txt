cmake_minimum_required(VERSION 3.10)
project(interactive_yolo_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(interactive_yolo_interfaces
  "msg/Bbox.msg"
  "msg/Annotation.msg"
  "msg/Prediction.msg"
  "msg/PredictionResult.msg"
  "msg/Point2D.msg"
  "msg/Segment.msg"
  "msg/Float32Tensor.msg"
  "msg/BoolTensor.msg"
  "msg/DatabaseAnnotationInfo.msg"
  "msg/DatabaseCategoryInfo.msg"
  "msg/DatabaseImageInfo.msg"
  "msg/DatabaseQuestionInfo.msg"
  "msg/DatabaseSolvedQuestionInfo.msg"
  "msg/DatabaseUpdateNotifaction.msg"
  "srv/database/GetDatabaseAnnotation.srv"
  "srv/database/GetDatabaseCategory.srv"
  "srv/database/GetDatabaseCategoryByName.srv"
  "srv/database/GetDatabaseImage.srv"
  "srv/database/GetDatabaseQuestion.srv"
  "srv/database/GetAllDatabaseCategories.srv"
  "srv/database/RegisterDatabaseAnnotation.srv"
  "srv/database/RegisterDatabaseCategory.srv"
  "srv/database/RegisterDatabaseImage.srv"
  "srv/database/RegisterDatabaseQuestion.srv"
  "srv/database/SetDatabaseAnnotationEmbedding.srv"
  "srv/database/SetDatabaseAnnotationMask.srv"
  "srv/database/SetDatabaseCategoryEmbeddings.srv"
  "srv/database/SetDatabaseCategoryZeroshotEmbedding.srv"
  "srv/database/RemoveDatabaseQuestion.srv"
  "srv/database/SolveDatabaseQuestion.srv"
  "srv/PonctualInference.srv"
  "action/UIAnnotation.action"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_package()
