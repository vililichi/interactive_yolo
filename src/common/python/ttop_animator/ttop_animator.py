from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage
import cv_bridge
import time
from .eyes import Eyes
import random
from threading import Lock, Thread

class AnimatorTTOP():

    def __init__(self, node, random_blink:bool = True):
        
        self.node = node
        self.pub_gesture = self.node.create_publisher(String, 'ttop_remote_proxy/gesture/name', 1)
        self.pub_face = self.node.create_publisher(CompressedImage, 'ttop_remote_proxy/display_input/compressed', 1)
        self.bridge = cv_bridge.CvBridge()

        self.lock = Lock()

        self.gesture_name = None
        self.emotion_img = None
        self.custom_img = None

        self.eyes = Eyes()
        self.eyes.eye_color = (255,255,255)
        self.eyes.skin_color = (0,0,0)
        self.actual_emotion = "sleep"

        self.eyes.set_emotion(self.actual_emotion)
        self.emotion_img = self.eyes.img
        self._update_robot()

        if random_blink:
            self.blink_thread = Thread(target=self._blink_loop, daemon=True)
            self.blink_thread.start()

    def _blink_loop(self):
        while True:
            self.blink()
            time.sleep(max(random.gauss(2.5, 1.0),0))
    
    def _update_robot(self):
        if self.gesture_name is not None:
            self.pub_gesture.publish(String(data=self.gesture_name))
            self.gesture_name = None
        
        if self.custom_img is not None:
            self.pub_face.publish(self.bridge.cv2_to_compressed_imgmsg(self.custom_img))
        elif self.emotion_img is not None:
            self.pub_face.publish(self.bridge.cv2_to_compressed_imgmsg(self.emotion_img))

    def set_custom_img(self, img):
        self.custom_img = img
        self._update_robot()
    
    def remove_custom_img(self):
        self.custom_img = None
        self._update_robot()

    def set_emotion(self, emotion:str):
        if self.custom_img is None:
            dt = 0.2
            nbr_itt = 5
            for i in range (nbr_itt):
                self.eyes.set_emotion_mix(self.actual_emotion, emotion, i/(nbr_itt-1))
                self.emotion_img = self.eyes.img
                self._update_robot()
                time.sleep(dt)
            self.actual_emotion = emotion

        else:
            self.eyes.set_emotion(emotion)
            self.emotion_img = self.eyes.img
            self._update_robot()
            self.actual_emotion = emotion

    def normal(self):
        with self.lock:
            self.gesture_name = "slow_origin_all"
            self.set_emotion("normal")

    def happy(self):
        with self.lock:
            self.gesture_name = "slow_origin_all"
            self.set_emotion("happy")

    def angry(self):
        with self.lock:
            self.gesture_name = "slow_origin_all"
            self.set_emotion("angry")

    def what(self):
        with self.lock:
            self.gesture_name = "showing"
            self.set_emotion("worried")

    def check_table(self):
        with self.lock:
            self.gesture_name = "sad"
            self.set_emotion("curious")

    def sleep(self):
        with self.lock:
            self.gesture_name = "sad"
            self.set_emotion("sleep")
    
    def sad(self):
        with self.lock:
            self.gesture_name = "sad"
            self.set_emotion("sad")
    
    def blink(self):
        with self.lock:
            if self.custom_img is None:
                dt = 0.1
                nbr_itt = 5
                for i in range (nbr_itt):
                    self.eyes.set_emotion_mix(self.actual_emotion, "sleep", i/(nbr_itt-1))
                    self.emotion_img = self.eyes.img
                    self._update_robot()
                    time.sleep(dt)
                time.sleep(0.1)
                for i in range (nbr_itt):
                    self.eyes.set_emotion_mix("sleep", self.actual_emotion, i/(nbr_itt-1))
                    self.emotion_img = self.eyes.img
                    self._update_robot()
                    time.sleep(dt)




