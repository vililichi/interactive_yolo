from std_msgs.msg import String, Bool
from queue import Queue

class SpeakListen():

    def __init__(self, node):
        self.sub_stt = node.create_subscription(String, 'interactive_yolo/stt', self._stt_callback, 10)
        self.pub_tts = node.create_publisher(String, 'interactive_yolo/tts', 10)
        self.pub_tts_reset = node.create_publisher(
            Bool,
            'interactive_yolo/tts_reset',
            10)
        self.listen_texts = Queue()

    def _stt_callback(self, msg:String):
        self.listen_texts.put(msg.data)

    def clear_listen_buffer(self):
        self.listen_texts = Queue()

    def clear_speak_buffer(self):
        msg = Bool()
        msg.data = True
        self.pub_tts_reset.publish(msg)

    def listen(self)->str:
        if self.listen_texts.empty():
            return ""
        else:
            return self.listen_texts.get()
    
    def speak(self, text:str):
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)
