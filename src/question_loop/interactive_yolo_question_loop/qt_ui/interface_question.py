from threading import Lock, Thread
import time

from .widget import QuestionWithImage, CaptureWidget
from PySide6.QtWidgets import QApplication, QStackedWidget
from PySide6 import QtCore

from speaklisten import SpeakListen
import yake

def extract_listen_answer(text:str):
    words = text.split(" ")
    description_words = ["c'est", "est", "sont"]
    validation_words = ["validé", "valider", "validez", "valide", "confirmer", "confirmez", "confirmé"]
    annulation_words = ["annuler", "annulez", "annulé"]
    
    description = False
    validation = False
    annulation = False
    object_name = None
    description_segment = ""
    for word in words:

        if word in validation_words:
            validation = True

        if word in annulation_words:
            annulation = True
        
        if word  in description_words:
            description = True
        
        if description:
            if description_segment == "":
                description_segment = word
            else:
                description_segment += " " + word

    if description:
        custom_kw_extractor = yake.KeywordExtractor(
            lan="fr",
            n=3,
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=1,
            top=10,
            features=None
        )
        keywords = custom_kw_extractor.extract_keywords(description_segment)
        potential_object = list()

        for word, _ in keywords:

            is_object = True

            if word in description_words:
                is_object = False
            
            for negative_filter in ["c'est",]:
                if negative_filter in word:
                    is_object = False
                    break
            
            if is_object:
                potential_object.append(word)
    
        if len(potential_object) > 0:
            object_name = potential_object[0]
    
    return object_name, validation, annulation

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
            if answer == "Objet invalide":
                self.answer = None
            else:
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

    def ask_question(self, cv_image, speak_listen:SpeakListen, estimation_label = None)->str:

        if estimation_label is not None:
            if estimation_label == "__NOTHING__":
                self.question_widget.setAnswer("Objet invalide")
            else:
                self.question_widget.setAnswer(estimation_label)

        self.question_widget.setImage(cv_image)
        self.answer = None
        self.answer_received = False

        self.stacked_widget.setCurrentIndex(0)

        speak_listen.speak("Quel est cet objet?")
        if estimation_label is not None and estimation_label != "":
            if estimation_label == "__NOTHING__":
                speak_listen.speak("Je crois que ce n'est pas un objet valide")
            else:
                speak_listen.speak("Je crois que c'est "+estimation_label)

        help_time = time.time() + 30
        answer = None
        speak_listen.clear_listen_buffer()
        while True:

            listen_text = speak_listen.listen()
            if listen_text != "":

                object_name, validation, annulation = extract_listen_answer(listen_text)
                if( object_name is not None and validation == False and annulation == False):
                    with self.answer_lock:
                        self.question_widget.setAnswer(object_name)
                        speak_listen.speak("Vous avez dit que l'objet est "+object_name+". Dites valider pour valider la réponse")
                    help_time += 10
                elif( object_name is None and validation == True and annulation == False):
                    self.question_widget.validButtonClicked()
                elif( object_name is None and validation == False and annulation == True):
                    self.question_widget.cancelButtonClicked()
                else:
                    if( help_time - time.time()) < 20:
                        speak_listen.speak("Désolé, je n'ai pas compris")
                    help_time -= 10

            with self.answer_lock:
                if self.answer_received:
                    answer = self.answer
                    if answer is None:
                        speak_listen.clear_speak_buffer()
                        speak_listen.speak("Objet invalidé")
                    else:
                        speak_listen.clear_speak_buffer()
                        speak_listen.speak("Objet validé, l'objet est "+answer)
                    break
                elif time.time() > help_time:
                    help_time = time.time() + 60
                    speak_listen.speak(text="Quel est cet objet?")
                    speak_listen.speak(text="Dites c'est un [nom de l'objet] pour donner le nom de l'objet")
                    speak_listen.speak(text="Dites [valider] pour valider le nom de l'objet")
                    speak_listen.speak(text="Dites [annuler] si ce n'est pas un objet")

            time.sleep(0.05)

        self.stacked_widget.setCurrentIndex(1)

        return answer