import sounddevice
import threading
import queue

import rclpy
import rclpy.node
from std_msgs.msg import String, Bool

import speech_recognition as sr
import json
import os

from urllib.request import urlretrieve
import zipfile

from interactive_yolo_utils import workspace_dir

import shutil
import time

VOSK_MODEL_PATH = os.path.join(workspace_dir(),"models","vosk")

def install_vosk_model():
    model_url= "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip"
    model_zip = "vosk-model-fr-0.22.zip"
    model_unzip_dir = "temporary_vosk_install"
    model_to_move = os.path.join(model_unzip_dir,"vosk-model-fr-0.22")

    # check if model dir exist
    if not os.path.isdir(VOSK_MODEL_PATH):

        #download model
        print("downloading vosk model")
        urlretrieve(model_url, model_zip)

        #unzip model
        print("unzip vosk model")
        with zipfile.ZipFile(model_zip,"r") as zip_ref:
            zip_ref.extractall(model_unzip_dir)

        print("install vosk model")
        shutil.move(model_to_move, VOSK_MODEL_PATH)

        print("cleaning")
        os.removedirs(model_unzip_dir)
        os.remove(model_zip)

        print("installation finished")

GOOGLE_API = 0
VOSK = 1
WHISPER = 2


class STTNode(rclpy.node.Node):
    def __init__(self, model_to_use = VOSK):
        super().__init__('stt_node')

        # Setup model
        self.model_to_use = model_to_use
        self.recognizer = sr.Recognizer()

        if self.model_to_use == GOOGLE_API:
            self.get_text = self.get_text_google_api
            self.model_name = "GOOGLE API"

        if self.model_to_use == VOSK :
            install_vosk_model()  
            from vosk import Model
            self.recognizer.vosk_model = Model(VOSK_MODEL_PATH)
            self.get_text = self.get_text_vosk
            self.model_name = "VOSK"

        if self.model_to_use == WHISPER :
            self.get_text = self.get_text_whisper
            self.model_name = "WHISPER"

        # Calibrate microphone
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        # Create publisher
        self.pub = self.create_publisher(String, 'interactive_yolo/stt', 10)

        # Start threads
        self.audio_queue = queue.Queue()

        self.thread_listen = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread_listen.start()

        self.thread_stt = threading.Thread(target=self._stt_loop, daemon=True)
        self.thread_stt.start()   

    def _listen_loop(self):
        
        while(True):

            with sr.Microphone() as source:

                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self.recognizer.listen(source)

                self.audio_queue.put(audio)

    def get_text_google_api(self, audio)->str:
        return self.recognizer.recognize_google(audio, language='fr-FR')
    
    def get_text_vosk(self, audio)->str:
        text = self.recognizer.recognize_vosk(audio, language='fr-FR')
        json_data = json.loads(text)
        return json_data.get('text', '')
    
    def get_text_whisper(self, audio)->str:
        return self.recognizer.recognize_whisper(audio, language='french', model="small")

    def _stt_loop(self):  

        while(True):

                audio = self.audio_queue.get()
                print("-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-")

                try:
                    text = self.get_text(audio)
                    print(self.model_name + " a entendu : " + text)
                    if text != "":
                        msg = String()
                        msg.data = text
                        self.pub.publish(msg)

                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(self.model_name + " erreur; {0}".format(e))    

def main(args=None):
    rclpy.init()

    node = STTNode(VOSK)
    print("Node ready")

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()