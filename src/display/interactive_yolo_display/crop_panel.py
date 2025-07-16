from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from rclpy.node import Node
import rclpy
from cv_bridge import CvBridge
from threading import Thread, Lock
import time
import sys
from .qt_utils.widget import ScaledImage

class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, label_text:str, start_at_max:bool = False):
        super().__init__()

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.precision = 1000
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.precision)
        if start_at_max:
            self.slider.setValue(self.precision)
        self.label = QtWidgets.QLabel(self, text=label_text)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout) 

        self.on_value_change = None

        self.slider.valueChanged.connect(self._on_value_changed)
    
    def _on_value_changed(self):

        value = self.slider.value()/self.precision

        if self.on_value_change is not None:
            self.on_value_change(value)


class CropPanel(QtWidgets.QWidget):
    def __init__(self, node:Node):
        super().__init__()

        self.node = node
        self._x1_publisher = self.node.create_publisher(Float32, 'interactive_yolo/crop_panel_output/x1',1)
        self._x2_publisher = self.node.create_publisher(Float32, 'interactive_yolo/crop_panel_output/x2',1)
        self._y1_publisher = self.node.create_publisher(Float32, 'interactive_yolo/crop_panel_output/y1',1)
        self._y2_publisher = self.node.create_publisher(Float32, 'interactive_yolo/crop_panel_output/y2',1)

        self._image = ScaledImage()

        self.slider_x1 = LabeledSlider("X1")
        self.slider_x1.on_value_change = self._x1_cb

        self.slider_x2 = LabeledSlider("X2", start_at_max=True)
        self.slider_x2.on_value_change = self._x2_cb

        self.slider_y1 = LabeledSlider("Y1")
        self.slider_y1.on_value_change = self._y1_cb

        self.slider_y2 = LabeledSlider("Y2", start_at_max=True)
        self.slider_y2.on_value_change = self._y2_cb

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._image, 4)
        layout.addWidget(self.slider_x1,1)
        layout.addWidget(self.slider_x2,1)
        layout.addWidget(self.slider_y1,1)
        layout.addWidget(self.slider_y2,1)
        self.setLayout(layout) 


    def setImage(self, cv_image):
        self._image.setImage(cv_image)

    def _x1_cb(self, value:float):
        msg = Float32()
        msg.data = value
        self._x1_publisher.publish(msg)

    def _x2_cb(self, value:float):
        msg = Float32()
        msg.data = value
        self._x2_publisher.publish(msg)

    def _y1_cb(self, value:float):
        msg = Float32()
        msg.data = value
        self._y1_publisher.publish(msg)

    def _y2_cb(self, value:float):
        msg = Float32()
        msg.data = value
        self._y2_publisher.publish(msg)


class CropPanelNode(Node):
    def __init__(self):
        super().__init__('display')

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        
        self.widget = CropPanel(self)
        self.widget.show()
        
        self.img_lock = Lock()
        self.cv_bridge = CvBridge()
        self.cv_image = None

        self.ttop_camera_raw_subscriber = self.create_subscription( Image, 'interactive_yolo/crop_panel_input', self.input_raw_cb, qos_profile=qos_policy)
        self.ttop_camera_raw_subscriber = self.create_subscription( CompressedImage, 'interactive_yolo/crop_panel_input_input_compressed', self.input_compressed_cb, qos_profile=qos_policy)

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

    node = CropPanelNode()

    def rclpy_thread_fun():
        rclpy.spin(node)
        rclpy.shutdown()

    rclpy_thread = Thread(target = rclpy_thread_fun, daemon=True)
    rclpy_thread.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()