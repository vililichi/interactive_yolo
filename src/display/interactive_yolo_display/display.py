from PyQt5 import QtWidgets
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
import rclpy
from cv_bridge import CvBridge
from threading import Thread, Lock
import time
import sys
from .qt_utils.widget import ScaledImage


class DisplayNode(Node):
    def __init__(self):
        super().__init__('display')

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        
        self.widget = ScaledImage()
        self.widget.show()
        
        self.img_lock = Lock()
        self.cv_bridge = CvBridge()
        self.cv_image = None

        self.ttop_camera_raw_subscriber = self.create_subscription( Image, 'interactive_yolo/display_input', self.input_raw_cb, qos_profile=qos_policy)
        self.ttop_camera_raw_subscriber = self.create_subscription( CompressedImage, 'interactive_yolo/display_input_compressed', self.input_compressed_cb, qos_profile=qos_policy)

        self.thread = Thread(target=self.rect_camera_loop, daemon=True)
        self.thread.start()

    def rect_camera_loop(self):

        while True:
            time.sleep(0.01)
            cv_image = None

            with self.img_lock:
                if( self.cv_image is not None ):
                    cv_image = self.cv_image
                    self.cv_image = None
                else:
                    continue

            self.widget.setImage(cv_image)

    def input_raw_cb(self, msg:Image):

        with self.img_lock:
            self.cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def input_compressed_cb(self, msg:CompressedImage):

        with self.img_lock:
            self.cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

def main(args=None):
    app = QtWidgets.QApplication([])
    rclpy.init()

    node = DisplayNode()

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()