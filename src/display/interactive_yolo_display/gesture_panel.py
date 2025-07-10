from PyQt5 import QtWidgets
from PyQt5.QtCore import QMetaObject, Qt
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
from threading import Thread, Lock
import time
import sys
from .qt_utils.widget import ConsoleWidget
from speaklisten import SpeakListenTTOP

class GestureControlWidget(QtWidgets.QWidget):
    def __init__(self, node:Node, gesture_list): 
        super().__init__()

        self.gesture_list = gesture_list
        self.node = node

        self.pub = self.node.create_publisher(String, 'ttop_remote_proxy/gesture/name', 1)
 
        # create objects
        self.buttons = []
        for gesture in gesture_list:
            self.node.get_logger().info(f'Gesture registered : {gesture}')
            button = QtWidgets.QPushButton(text = gesture, parent = self)
            button.clicked.connect(lambda command = gesture: self._button_cb(command))
            self.buttons.append(button)


        # layout
        layout = QtWidgets.QVBoxLayout(self)
        for button in self.buttons:
            layout.addWidget(button)
        self.setLayout(layout) 

    def _button_cb(self, text):
        msg = String()
        msg.data = text
        self.pub(msg)

class GesturePanelNode(Node):
    def __init__(self):
        super().__init__('SpeakConsoleNode')

        self.gesture_list = [
            'yes',
            'no',
            'maybe',
            'origin_all',
            'origin_all',
            'origin_head',
            'origin_torso',
            'thinking',
            'sad',
            'check_table'
        ]

        self.widget = GestureControlWidget(self, gesture_list=self.gesture_list)
        self.widget.show()

def main(args=None):
    app = QtWidgets.QApplication([])
    rclpy.init()

    node = GesturePanelNode()

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
