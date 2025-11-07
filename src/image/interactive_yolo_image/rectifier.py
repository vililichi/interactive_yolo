import cv2
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from rclpy.node import Node
import numpy as np
import rclpy
from cv_bridge import CvBridge
from threading import Thread, Lock
import time

class RectifierNode(Node):
    def __init__(self):
        super().__init__('rectifier')

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        
        self.map_lock = Lock()
        self.img_lock = Lock()

        self.cv_bridge = CvBridge()
        self.mapx, self.mapy = None, None
        self.cv_images_list = []

        # Setup model
        self.pub = self.create_publisher(Image, 'interactive_yolo/image_rect', qos_profile=qos_policy)
        self.pub_compressed = self.create_publisher(CompressedImage, 'interactive_yolo/image_rect/compressed', qos_profile=qos_policy)

        self.ttop_camera_raw_subscriber = self.create_subscription( Image, 'interactive_yolo/rect_camera_raw', self.rect_camera_raw_cb, qos_profile=qos_policy)
        self.ttop_camera_raw_subscriber = self.create_subscription( CompressedImage, 'interactive_yolo/rect_camera_compressed', self.rect_camera_compressed_cb, qos_profile=qos_policy)
        self.ttop_camera_info_subscriber = self.create_subscription( CameraInfo, 'interactive_yolo/rect_camera_info', self.rect_camera_info_cb, qos_profile=qos_policy)

        self.thread = Thread(target=self.rect_camera_loop, daemon=True)
        self.thread.start()
    
    def add_image(self, cv_image)->bool:
        with self.img_lock:
            self.cv_images_list.append(cv_image)
            if len(self.cv_images_list) > 5:
                self.cv_images_list = self.cv_images_list[-5:]
    
    def get_image(self):
        while True:
            with self.img_lock:
                if len(self.cv_images_list) > 0:
                    return self.cv_images_list.pop(0)
            time.sleep(0.01)

    def rect_camera_loop(self):

        mapx = None
        mapy = None
        while True:
            time.sleep(1.0)
            with self.map_lock:
                if( self.mapx is not None and self.mapy is not None ):
                    mapx = self.mapx
                    mapy = self.mapy
                    break
        
        while True:
            cv_image = self.get_image()
            cv_image = cv2.remap(cv_image, mapx, mapy, cv2.INTER_LINEAR)
            self.pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.pub_compressed.publish(self.cv_bridge.cv2_to_compressed_imgmsg(cv_image, "jpg"))


    def rect_camera_raw_cb(self, msg:Image):

        self.add_image(self.cv_bridge.imgmsg_to_cv2(msg, "bgr8"))

    def rect_camera_compressed_cb(self, msg:CompressedImage):

        self.add_image(self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8"))

    def rect_camera_info_cb(self, msg:CameraInfo):

        with self.map_lock:
            if( self.mapx is None and self.mapy is None ):
                self.cameraMatrix = np.array(msg.k).reshape((3, 3))
                self.distortion = np.array(msg.d)
                self.rectify = np.array(msg.r).reshape((3, 3))
                self.w = msg.width
                self.h = msg.height
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.cameraMatrix, self.distortion, self.rectify, None, (self.w, self.h), cv2.CV_32FC1)

def main(args=None):
    rclpy.init()

    node = RectifierNode()
    print("Node ready")

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()