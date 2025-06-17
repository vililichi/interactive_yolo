from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='interactive_yolo_database',
            namespace='interactive_yolo',
            executable='database_node',
            name='database'
        ),
        Node(
            package='interactive_yolo_model',
            namespace='interactive_yolo',
            executable='model_node',
            name='model'
        ),
        Node(
            package='interactive_yolo_question_loop',
            namespace='interactive_yolo',
            executable='question_loop_node',
            name='question_loop'
        ),
    ])
