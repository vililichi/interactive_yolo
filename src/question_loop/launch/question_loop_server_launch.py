from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='interactive_yolo_database',
            executable='database_node',
            name='database'
        ),
        Node(
            package='interactive_yolo_model',
            executable='model_node',
            name='model'
        )
    ])
