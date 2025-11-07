from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image as sensor_msgs_Image
from std_msgs.msg import Bool
from threading import Lock
import time

class Camera():

    def __init__(self, node:Node, topic:str, image_request_topic:str = None):
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

        if image_request_topic is None:
            self._image_request_publisher = None
        else:
            self._image_request_publisher = self.node.create_publisher(Bool, image_request_topic,1)

    def request_image(self):
        if self._image_request_publisher is None:
            return
        
        msg = Bool()
        msg.data = True
        self._image_request_publisher.publish(msg)

    def capture(self):
        with self.lock:
            self.cv_image = None
            self.image_wanted = True

        self.request_image()
        nbr_try = 0
        
        while True:
            time.sleep(0.05)
            with self.lock:

                if self.cv_image is not None:
                    return self.cv_image.copy()
                
                if self.image_wanted == False:
                    return None
                
                if nbr_try >= 20:
                    nbr_try = 0
                    self.request_image()
                    
                else:
                    nbr_try += 1
        

    def img_callback(self, msg):
        with self.lock:
            if self.image_wanted:
                self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.image_wanted = False