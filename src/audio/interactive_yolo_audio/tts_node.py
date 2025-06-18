from gtts import gTTS
from threading import Lock, Thread
from queue import Queue
import os

import rclpy
import rclpy.node
from std_msgs.msg import String, Bool
from playsound import playsound
from interactive_yolo_utils import workspace_dir

class TTSNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('stt_node')

        self.language = 'fr'
        self.tld = "fr"

        self.speak_lock = Lock()

        self.audio_dir = os.path.join(workspace_dir(), "audio_sample")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.audio_dict = dict()

        self.speak_queue = Queue()
        self.stop = False

        self.speak_thread = Thread(target = self.speak_loop, daemon=True)
        self.speak_thread.start()

        self.sub_tts = self.create_subscription(
            String,
            'interactive_yolo/tts',
            self._sub_tts_callback,
            10)
        
        self.pub_tts_speak = self.create_publisher(
            Bool,
            'interactive_yolo/tts_speak',
            10)
        
        self.sub_tts_reset = self.create_subscription(
            Bool,
            'interactive_yolo/tts_reset',
            self._sub_tts_reset_callback,
            10)

    def _sub_tts_callback(self, msg:String):
            self.register_speak(msg.data)

    def _sub_tts_reset_callback(self, msg:Bool):
            if msg.data:
                self.speak_thread
                self.speak_queue = Queue()

    def register_speak(self, text):
        self.speak_queue.put(text)

    def speak_loop(self):

        while True:

            text = None
            try:
                text = self.speak_queue.get(timeout=0.5)
            except:
                text = None

            if self.stop:
                break

            if text is None:
                continue

            self.speak(text)

    def speak(self, text:str):

        with self.speak_lock:

            if text not in self.audio_dict.keys():
                myobj = gTTS(text=text, lang=self.language, tld=self.tld, slow=False)
                file_name = str(len(self.audio_dict.keys())).rjust(4,"0")+".mp3"
                file_path = os.path.join(self.audio_dir, file_name)
                myobj.save(file_path)
                self.audio_dict[text] = file_path

            yes_msg = Bool()
            no_msg = Bool()
            yes_msg.data = True
            no_msg.data = False
            self.pub_tts_speak.publish(yes_msg)
            print(" start speaking : ", text)
            playsound(self.audio_dict[text])
            self.pub_tts_speak.publish(no_msg)
            print(" end speaking ")

    def clean(self):
        with self.speak_lock:

            for path in self.audio_dict.values():
                os.remove(path)
            self.audio_dict = dict()


def main(args=None):
    try:
        rclpy.init()
        node = TTSNode()
        print("Node ready")
        rclpy.spin(node)
        rclpy.shutdown()
        

    finally:
        node.stop = True
        node.clean()


if __name__ == '__main__':
    main()