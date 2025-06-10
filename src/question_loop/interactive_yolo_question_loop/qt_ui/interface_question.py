from threading import Lock, Thread
import time

from .widget import QuestionWithImage, CaptureWidget
from PySide6.QtWidgets import QApplication, QStackedWidget
from PySide6 import QtCore

class interface_question:

    def __init__(self):

        self.question_widget = QuestionWithImage()
        self.question_widget.setQuestion("Quel est cet objet?")
        self.question_widget.setAnswer("Truc")
        self.question_widget.setValidButtonLabel("Valider")
        self.question_widget.setCancelButtonLabel("Pas un objet")

        self.question_widget.setValidCallback(self.valid_callback)
        self.question_widget.setCancelCallback(self.cancel_callback)

        self.capture_widget = CaptureWidget()
        self.capture_widget.setSwapButtonMode1Name("Image brute")
        self.capture_widget.setSwapButtonMode2Name("Image annotée")

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.question_widget)
        self.stacked_widget.addWidget(self.capture_widget)

        self.answer = None
        self.answer_received = False
        self.answer_lock = Lock()

        self.stacked_widget.setCurrentIndex(1)
        self.stacked_widget.show()

    def valid_callback(self, answer):
        with self.answer_lock:
            self.answer = answer
            self.answer_received = True

    def cancel_callback(self):
        with self.answer_lock:
            self.answer = None
            self.answer_received = True

    def set_capture_image_brute(self, cv_image):
        self.capture_widget.setImage1(cv_image)

    def set_capture_image_annotee(self, cv_image):
        self.capture_widget.setImage2(cv_image)

    def set_capture_callback(self, callback):
        self.capture_widget.setCaptureCallback(callback)

    def ask_question(self, cv_image)->str:

        self.question_widget.setImage(cv_image)
        self.answer = None
        self.answer_received = False

        self.stacked_widget.setCurrentIndex(0)

        answer = None
        while True:
            with self.answer_lock:
                if self.answer_received:
                    answer = self.answer
                    break
            time.sleep(0.05)

        self.stacked_widget.setCurrentIndex(1)

        return answer