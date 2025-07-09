from PyQt5 import QtWidgets
from PyQt5.QtCore import QMetaObject, Qt
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
import rclpy
from threading import Thread, Lock
import time
import sys
from .qt_utils.widget import ConsoleWidget
from speaklisten import SpeakListenTTOP

class SpeakConsoleNode(Node):
    def __init__(self):
        super().__init__('SpeakConsoleNode')
        self.console_widget = ConsoleWidget()

        self.log_list = []
        self.log_lock = Lock()
        
        self.speak_listen_ttop = SpeakListenTTOP(self)

        self.console_widget.input_callback = self.input_callback
        self.console_widget.show()

        self.listen_thread = Thread(target=self.listen_loop, daemon=True)
        self.listen_thread.start()

        self.speak_listen_ttop.talk_start_cb = self.speak_start_cb
        self.speak_listen_ttop.talk_end_cb = self.talk_end_cb
    
    def speak_start_cb(self):
        with self.log_lock:
            self.log_list.append(("SYS", "T-Top start talking"))

    def talk_end_cb(self):
        with self.log_lock:
            self.log_list.append(("SYS", "T-Top end talking"))

    def input_callback(self, text):
        self.speak_listen_ttop.wait_talking_end()
        self.speak_listen_ttop.speak(text)
        with self.log_lock:
            self.log_list.append(("SPK", text))
            

    def listen_loop(self):
        while True:
            time.sleep(0.5)
            text = self.speak_listen_ttop.listen()
            if text != "":
                with self.log_lock:
                    self.log_list.append(("LIS", text))
            self.update_console()

    def update_console(self):
        with self.log_lock:
            for usr, msg in self.log_list:
                self.console_widget.log(usr, msg)
            self.log_list.clear()



        


def main(args=None):
    app = QtWidgets.QApplication([])
    rclpy.init()

    node = SpeakConsoleNode()

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
