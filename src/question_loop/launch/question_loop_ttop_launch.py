from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='interactive_yolo_question_loop',
            namespace='interactive_yolo',
            executable='question_loop_node',
            name='question_loop',
            parameters=[
                {'image_sending_mode': 'compressed'},
                {'input_mode':'ttop'}
            ]
        ),
        Node(
              package="cv_camera",
              namespace='interactive_yolo',
              executable='cv_camera_node',
              name='cv_camera_node',
              parameters=[
                  {"rate": 15.0},
                  {"device_path": "/dev/camera_2d_wide"},
                  {"image_width": 1280},
                  {"image_height": 720},
              ],
              remappings=[
                  ("cv_camera_node/image_raw", "interactive_yolo/image_raw"),
              ]
        )
    ])