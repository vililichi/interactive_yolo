from PyQt5 import QtWidgets
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
from threading import Thread
import sys
from ttop_animator import AnimatorTTOP

class GestureControlWidget(QtWidgets.QWidget):
    def __init__(self, node:Node): 
        super().__init__()

        self.node = node

        self.animator = AnimatorTTOP(node)

        # create objects
        self.buttons = []

        button = QtWidgets.QPushButton(text = "happy", parent = self)
        button.clicked.connect(self.animator.happy)
        self.buttons.append(button)

        button = QtWidgets.QPushButton(text = "angry", parent = self)
        button.clicked.connect(self.animator.angry)
        self.buttons.append(button)

        button = QtWidgets.QPushButton(text = "what", parent = self)
        button.clicked.connect(self.animator.what)
        self.buttons.append(button)

        button = QtWidgets.QPushButton(text = "check_table", parent = self)
        button.clicked.connect(self.animator.check_table)
        self.buttons.append(button)

        button = QtWidgets.QPushButton(text = "sad", parent = self)
        button.clicked.connect(self.animator.sad)
        self.buttons.append(button)

        button = QtWidgets.QPushButton(text = "sleep", parent = self)
        button.clicked.connect(self.animator.sleep)
        self.buttons.append(button)

        button = QtWidgets.QPushButton(text = "wink", parent = self)
        button.clicked.connect(self.animator.wink)
        self.buttons.append(button)


        # layout
        layout = QtWidgets.QVBoxLayout(self)
        for button in self.buttons:
            layout.addWidget(button)
        self.setLayout(layout) 

class GesturePanelNode(Node):
    def __init__(self):
        super().__init__('SpeakConsoleNode')

        self.widget = GestureControlWidget(self)
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
