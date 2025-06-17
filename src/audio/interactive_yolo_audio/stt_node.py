import sounddevice
import threading
import queue

import rclpy
import rclpy.node
from std_msgs.msg import String

import speech_recognition as sr
import json


class STTNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('stt_node')

        self.pub = self.create_publisher(String, 'interactive_yolo/stt', 10)

        self.recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        self.audio_queue = queue.Queue()

        self.thread_listen = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread_listen.start()

        self.thread_stt = threading.Thread(target=self._stt_loop, daemon=True)
        self.thread_stt.start()

    def _listen_loop(self):
        
        while(True):

            with sr.Microphone() as source:
                audio = self.recognizer.listen(source)
                self.audio_queue.put(audio)

    def _stt_loop(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        while(True):

                
                audio = self.audio_queue.get()
                print("-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-")

                try:
                    text = recognizer.recognize_google(audio, language='fr-FR')
                    if text != "":
                        print("google thinks you said " + text)
                        msg = String()
                        msg.data = text
                        self.pub.publish(msg)
                        
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print("google error; {0}".format(e))

                continue

                try:
                    text = recognizer.recognize_vosk(audio, language='fr-FR')
                    json_data = json.loads(text)
                    text = json_data.get('text', '')
                    if text != "":
                        print("vosk thinks you said " + text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print("vosk error; {0}".format(e))

                try:
                    text = recognizer.recognize_whisper(audio, language='french')
                    if text != "":
                        print("whisper thinks you said " + text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print("whisper error; {0}".format(e))


                

    


def main(args=None):
    rclpy.init()

    node = STTNode()
    print("Node ready")

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()