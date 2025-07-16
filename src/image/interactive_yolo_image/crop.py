from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from rclpy.node import Node
import numpy as np
import rclpy
from cv_bridge import CvBridge
from threading import Thread

class CropNode(Node):
    def __init__(self):
        super().__init__('crop')

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        
        self.x1 = 0.0
        self.x2 = 1.0
        self.y1 = 0.0
        self.y2 = 1.0

        self.cv_bridge = CvBridge()
        self.mapx, self.mapy = None, None
        self.cv_image = None

        # Setup model
        self.pub = self.create_publisher(Image, 'interactive_yolo/image_cropped', qos_profile=qos_policy)
        self.pub_compressed = self.create_publisher(CompressedImage, 'interactive_yolo/image_cropped/compressed', qos_profile=qos_policy)

        self.ttop_camera_raw_subscriber = self.create_subscription( Image, 'interactive_yolo/image_to_crop', self.rect_camera_raw_cb, qos_profile=qos_policy)
        self.ttop_camera_raw_subscriber = self.create_subscription( CompressedImage, 'interactive_yolo/compressed_image_to_crop', self.rect_camera_compressed_cb, qos_profile=qos_policy)

        self.x1_subscriber = self.create_subscription( Float32, 'interactive_yolo/crop/x1', self.x1_callback, 1)
        self.x2_subscriber = self.create_subscription( Float32, 'interactive_yolo/crop/x2', self.x2_callback, 1)
        self.y1_subscriber = self.create_subscription( Float32, 'interactive_yolo/crop/y1', self.y1_callback, 1)
        self.y2_subscriber = self.create_subscription( Float32, 'interactive_yolo/crop/y2', self.y2_callback, 1)


    def process_img(self, cv_image:np.ndarray):

        h, w, c = cv_image.shape

        h1      = int(max(min(self.y1*h, h-1),0))
        h2      = int(max(min(self.y2*h, h-1),0))
        w1      = int(max(min(self.x1*w, w-1),0))
        w2      = int(max(min(self.x2*w, w-1),0))

        h_min = min(h1, h2)
        h_max = max(h1, h2)
        w_min = min(w1, w2)
        w_max = max(w1, w2)

        out_image = cv_image[h_min:h_max, w_min:w_max, :]

        self.pub.publish(self.cv_bridge.cv2_to_imgmsg(out_image, "bgr8"))
        self.pub_compressed.publish(self.cv_bridge.cv2_to_compressed_imgmsg(out_image, "jpg"))

    def rect_camera_raw_cb(self, msg:Image):
        self.process_img(self.cv_bridge.imgmsg_to_cv2(msg, "bgr8"))

    def rect_camera_compressed_cb(self, msg:CompressedImage):
        self.process_img(self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8"))
    
    def x1_callback(self, msg:Float32):
        self.x1 = msg.data

    def x2_callback(self, msg:Float32):
        self.x2 = msg.data

    def y1_callback(self, msg:Float32):
        self.y1 = msg.data

    def y2_callback(self, msg:Float32):
        self.y2 = msg.data



def main(args=None):
    rclpy.init()

    node = CropNode()
    print("Node ready")

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()