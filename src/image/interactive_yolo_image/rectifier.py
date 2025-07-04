import cv2
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from rclpy.node import Node
import numpy as np
import rclpy
from cv_bridge import CvBridge

class RectifierNode(Node):
    def __init__(self):
        super().__init__('rectifier')

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)

        self.cv_bridge = CvBridge()
        self.mapx, self.mapy = None, None

        # Setup model
        self.pub = self.create_publisher(Image, 'interactive_yolo/image_rect', qos_profile=qos_policy)
        self.pub_compressed = self.create_publisher(CompressedImage, 'interactive_yolo/image_rect/compressed', qos_profile=qos_policy)

        self.ttop_camera_raw_subscriber = self.create_subscription( Image, 'interactive_yolo/rect_camera_raw', self.rect_camera_raw_cb, qos_profile=qos_policy)
        self.ttop_camera_raw_subscriber = self.create_subscription( CompressedImage, 'interactive_yolo/rect_camera_compressed', self.rect_camera_compressed_cb, qos_profile=qos_policy)
        self.ttop_camera_info_subscriber = self.create_subscription( CameraInfo, 'interactive_yolo/rect_camera_info', self.rect_camera_info_cb, qos_profile=qos_policy)

    def rect_camera_raw_cb(self, msg:Image):

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        if( self.mapx is not None and self.mapy is not None ):
            cv_image = cv2.remap(cv_image, self.mapx, self.mapy, cv2.INTER_LINEAR)
            self.pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.pub_compressed.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def rect_camera_compressed_cb(self, msg:CompressedImage):

        cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if( self.mapx is not None and self.mapy is not None ):
            cv_image = cv2.remap(cv_image, self.mapx, self.mapy, cv2.INTER_LINEAR)
            self.pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.pub_compressed.publish(self.cv_bridge.cv2_to_compressed_imgmsg(cv_image, "jpg"))

    def rect_camera_info_cb(self, msg:CameraInfo):

        if( self.mapx is not None and self.mapy is not None ):
            pass

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