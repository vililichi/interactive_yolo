from std_msgs.msg import String, Bool
from queue import Queue
import time

class SpeakListenTTOP():

    def __init__(self, node):
        self.listen_texts = Queue()
        self._is_talking = False
        self._is_listening = False

        self.sub_stt = node.create_subscription(String, 'ttop_remote_proxy/stt', self._stt_callback, 1)
        self.sub_is_talking = node.create_subscription(Bool, "ttop_remote_proxy/is_talking", self._is_talking_callback, 1)
        self.sub_is_listening = node.create_subscription(Bool, "ttop_remote_proxy/is_listening", self._is_listening_callback, 1)
        self.pub_tts = node.create_publisher(String, 'ttop_remote_proxy/tts', 1)
        self.pub_tts_start = node.create_publisher(Bool, 'ttop_remote_proxy/start_stt', 1)

        self.talk_start_cb = None
        self.talk_end_cb = None

        self.listen_start_cb = None
        self.listen_end_cb = None

    def _is_talking_callback(self, msg:Bool):
        self._is_talking = msg.data
        if msg.data and self.talk_start_cb:
            self.talk_start_cb()
        if not msg.data and self.talk_end_cb:
            self.talk_end_cb()

    def _is_listening_callback(self, msg:Bool):
        self._is_listening = msg.data
        if msg.data and self.listen_start_cb:
            self.listen_start_cb()
        if not msg.data and self.listen_end_cb:
            self.listen_end_cb()

    def _stt_callback(self, msg:String):
        self.listen_texts.put(msg.data)

    def start_listen(self):
        msg = Bool()
        msg.data = True
        self.pub_tts_start.publish(msg)

    def get_listen(self)->str:
        if self.listen_texts.empty():
            return ""
        else:
            return self.listen_texts.get()
    
    def speak(self, text:str):
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)

    def is_talking(self) -> bool:
        return self._is_talking
    
    def wait_talking_end(self):
        while self._is_talking:
            time.sleep(0.1)

    def is_listening(self) -> bool:
        return self._is_listening
    
    def wait_listening_end(self):
        while self._is_listening:
            time.sleep(0.1)
