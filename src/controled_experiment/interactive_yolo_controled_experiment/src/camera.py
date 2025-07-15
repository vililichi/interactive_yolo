from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image as sensor_msgs_Image
from threading import Lock
import time

class Camera():

    def __init__(self, node:Node, topic:str):
        self.node = node
        self.bridge = CvBridge()
        self.topic = topic

        self.image_wanted = False
        self.cv_image = None
        self.lock = Lock()

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                depth=1)
        
        self.subscription = self.node.create_subscription(sensor_msgs_Image, self.topic, self.img_callback, qos_profile=qos_policy)

    def capture(self):
        with self.lock:
            self.cv_image = None
            self.image_wanted = True
        
        while True:
            time.sleep(0.05)
            with self.lock:

                if self.cv_image is not None:
                    return self.cv_image.copy()
                
                if self.image_wanted == False:
                    return None
        

    def img_callback(self, msg):
        with self.lock:
            if self.image_wanted:
                self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.image_wanted = False