import rclpy
from rclpy.node import Node

from speaklisten import SpeakListenTTOP
from llm_interpreter import LLMInterpreter
from ttop_animator import AnimatorTTOP
from .src.camera import Camera
from .src.model import Model
from .src.question_sorting import sort_questions, question_filter, question_nms
from .src.question_presentation import generate_question_presentation

import random

from threading import Thread

import time
import cv2
import numpy as np

from typing import List

STATE_SLEEP = 0
STATE_CONFIRM_START = 1
STATE_START = 2
STATE_CONFIRM_END = 3
STATE_END = 4
STATE_CALIBRATION = 5
STATE_IMAGE_CAPTURE = 6
STATE_IMAGE_VALIDATION = 7
STATE_IMAGE_ANALYSIS = 8
STATE_QUESTION = 9
STATE_VALIDATE_QUESTION = 10
STATE_SAVE_ANSWER = 11

objects_classes = [
    "humain",
    "meuble",
    "ustensiles",
    "vaisselle",
    "sandwich",
    "pâtes",
    "tarte",
    "salade",
    "pizza",
    "soupe",
    "boisson",
    "riz",
    "patate",
    "fromage",
    "pain",
    "oeuf",
    "poisson",
    "viande",
    "fruit",
    "légume",
    "dessert",
    "autre plat principal",
    "autre accompagnement"
]

objects_choices = objects_classes + [
    "objet invalide",
    "aucune de ces réponses"
]

class Question():
    def __init__(self):
        pass

class Experiment_node(Node):

    def __init__(self):
        super().__init__('experiment_orchestrator_node')

        self.state = STATE_SLEEP
        self.sub_state = 0
        self.last_state = (STATE_SLEEP, 0)

        self.speak_listen = SpeakListenTTOP(self)
        self.llm_interpreter = LLMInterpreter()
        self.animator = AnimatorTTOP(self)
        self.camera = Camera(self, 'interactive_yolo/experiment_image_input')

        self.listen_off()

        self.model = Model(classes_list=objects_classes)

        self.object_image = None

        self.nbr_question = 0
        self.remaining_question : List[Question] = []

        self.state_machine_thread = Thread(target=self.state_machine_loop, daemon=True)
        self.state_machine_thread.start()

    def state_machine_loop(self):
        while True:
            if self.state == STATE_SLEEP:
                self.sleep_state()
            elif self.state == STATE_CONFIRM_START:
                self.confirm_start_state()
            elif self.state == STATE_START:
                self.start_state()
            elif self.state == STATE_CONFIRM_END:
                self.confirm_end_state()
            elif self.state == STATE_END:
                self.end_state()
            elif self.state == STATE_CALIBRATION:
                self.calibration_state()
            elif self.state == STATE_IMAGE_CAPTURE:
                self.image_capture_state()
            elif self.state == STATE_IMAGE_VALIDATION:
                self.image_validation()
            elif self.state == STATE_IMAGE_ANALYSIS:
                self.image_analysis()
            elif self.state == STATE_QUESTION:
                self.question_state()
            elif self.state == STATE_VALIDATE_QUESTION:
                self.validate_question_state()
            elif self.state == STATE_SAVE_ANSWER:
                self.save_answer_state()
            else:
                raise ValueError(f"Invalid state: {self.state}")
            time.sleep(0.1) 

    def set_state(self, state):
        self.last_state = (self.state, self.sub_state)
        self.state = state
        self.sub_state = 0
        self.get_logger().info("Set state: "+str(self.state))

    def return_to_last_state(self):
        self.state, self.sub_state = self.last_state
        self.get_logger().info("Return to last state: "+str(self.state))

    def check_for_experiment_end(self, user_input)->bool:
        if user_input == "":
            return False
        
        text = "An IA and an user are doing an experiment where the user try to teach something to the IA."
        text +="\nDuring the dialog, the user says \""+user_input+"\""
        answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user want to end the experiment?", text, yes_threshold = 0.8)
        if answer:
            self.set_state(STATE_CONFIRM_END)
        return answer

    def listen_on(self):
        if not self.speak_listen.is_listening() and not self.speak_listen.is_talking():
            self.speak_listen.start_listen()
    
    def listen_off(self):

        self.speak_listen.stop_listen()
        while not self.speak_listen.listen_texts.empty():
            self.speak_listen.listen_texts.get()

    def try_listen(self):
       return self.speak_listen.get_listen()
    
    def speak(self, text):
        self.speak_listen.speak(text)
        time.sleep(0.5)
        while self.speak_listen.is_talking():
            time.sleep(0.1)
        return

    def sleep_state(self):

        if not self.animator.actual_emotion == "sleep":
            self.animator.sleep()

        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            text = """A scientist talk to an user to explain an experiment with a robot named T-Top.
            T-Top is waiting for a signal to start the experiment.
            The signal always containt the word "expérience"."""
            text +="\nThe robot hear \""+user_input+"\""
            text += """
            The robot hesitate to start the experiment because it does not know if the message is for him.
            The majority of the messages are not for him.
            """
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do T-Top received the signal to start the experiment?", text, yes_threshold=0.8)
            if answer:
                self.set_state(STATE_CONFIRM_START)
                return

            text = """A scientist talk to an user to explain an experiment with a robot named T-Top.
            T-Top is waiting for a signal to start the calibration.
            The signal always containt the word calibration."""
            text +="\nThe robot hear \""+user_input+"\""
            text += """
            The robot hesitate to start the calibration because it does not know if the message is for him.
            The majority of the messages are not for him.
            """
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do T-Top received the signal to start the calibration?", text, yes_threshold=0.8)
            if answer:
                self.set_state(STATE_CALIBRATION)
                return

    def confirm_start_state(self):

        if self.sub_state == 0:
            self.listen_off()
            self.animator.happy()
            self.speak("Bonjour!")
            self.animator.normal()
            self.speak("Voulez-vous commencez l'expérience?")
            self.sub_state = 1
            return
        
        if not self.animator.actual_emotion == "normal":
            self.animator.normal()
        
        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            text = "An IA ask to the user \"Voulez-vous commencez l'expérience?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user validate?", text, yes_threshold = 0.5)

            if answer:
                self.set_state(STATE_IMAGE_CAPTURE)
            else:
                self.listen_off()
                self.speak("J'ai compris que bous ne voulez pas commencer l'expérience.")
                self.speak("Je vais donc me rendormir.")
                self.set_state(STATE_SLEEP)
        
    def confirm_end_state(self):
        
        if self.sub_state == 0:
            self.listen_off()
            self.animator.sad()
            self.speak("Jai l'impression que vous voulez arrêter l'expérience.")
            self.animator.normal()
            self.speak("Est-ce que j'ai raison?")
            self.sub_state = 1
            return
        
        if not self.animator.actual_emotion == "normal":
            self.animator.normal()
        
        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            text = "An IA ask to the user \"Jai l'impression que vous voulez arrêter l'expérience.Est-ce que j'ai raison?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user want to end the experiment?", text, yes_threshold = 0.5)

            if answer:
                self.set_state(STATE_END)
            else:
                self.listen_off()
                self.animator.happy()
                self.speak("Alors, nous allons continuer l'expérience.")
                self.animator.normal()
                self.return_to_last_state()

    def start_state(self):

        if len(self.questions) > 0:
            self.set_state(STATE_QUESTION)
        else:
            self.listen_off()
            self.animator.happy()
            self.speak("J'ai terminé de poser toutes mes questions.")
            self.set_state(STATE_END)
        
    def end_state(self):

        self.listen_off()
        self.animator.happy()
        self.speak("Merci pour votre participation à l'expérience.")
        self.set_state(STATE_SLEEP)

    def calibration_state(self):
        
        if self.sub_state == 0:
            self.listen_off()
            self.sub_state = 1
            self.animator.happy()
            self.speak("Je me met en position pour la calibration.")
            self.speak("Donnez moi un signal lorsque la calibration est terminée.")
            self.animator.check_table()
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            return
        
        if user_input != "":
            text = """A user is doing an experiment with a robot named T-Top.
            T-Top is waiting for a signal to take a picture.
            T-Top said \"Je me met en position pour la calibration. Donnez moi un signal lorsque la calibration est terminée.\""""
            text +="\nThe user said \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Has T-Top received the signal indicating the end of the calibration?", text, yes_threshold=0.7)
            if answer:
                self.listen_off()
                self.speak("Fin de la calibration")
                self.set_state(STATE_SLEEP)
                return

    def image_capture_state(self):

        if self.sub_state == 0:
            self.listen_off()
            self.sub_state = 1
            self.animator.happy()
            self.speak("Parfait, commençons l'expérience.")
            self.animator.normal()
            self.speak("D'abord, placez votre repas sur le centre de la table et donnez moi un signal pour que je prenne une photo.")
            return
        
        if self.sub_state == 2:
            self.listen_off()
            self.sub_state = 1
            self.animator.normal()
            self.speak("Nous allons alors recommencer la photo.")
            self.animator.normal()
            self.speak("Placez votre repas sur le centre de la table et donnez moi un signal pour que je prenne une photo.")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            return
        
        if user_input != "":
            text = """A user is doing an experiment with a robot named T-Top.
            T-Top is waiting for a signal to take a picture.
            T-Top said \"Placez votre repas sur le centre de la table et donnez moi un signal pour que je prenne une photo.\""""
            text +="\nThe user said \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do T-Top received the signal to take a picture?", text, yes_threshold=0.7)
            if answer:
                self.listen_off()
                self.animator.check_table()
                self.speak("Cheeeeeeeeeeese")
                self.object_image = self.camera.capture()
                self.speak("Image capturée.")

                self.set_state(STATE_IMAGE_VALIDATION)
                return

    def image_validation(self):

        if self.sub_state == 0:
            self.listen_off()
            self.sub_state = 1
            self.animator.show()
            if self.object_image is None:
                self.get_logger().info("Trying to send None image")
            self.animator.set_custom_img(self.object_image)
            self.speak("Voici la photo que j'ai prise.")
            self.speak("La photo est-elle adéquate?")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            text = "An IA ask to the user \"Voici la photo que j'ai prise. La photo est-elle adéquate?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user validate the photo?", text, yes_threshold = 0.7)

            self.animator.remove_custom_img()
            if answer:
                self.set_state(STATE_IMAGE_ANALYSIS)
            else:
                self.set_state(STATE_IMAGE_CAPTURE)
                self.sub_state = 2
                
    def image_analysis(self):
        
        self.listen_off()
        self.animator.normal()
        self.speak_listen.speak("J'analyse actuellement la photo afin de trouver des questions qui me permetteront de mieux la comprendre.")
    
        questions = self.model.generate_question(self.object_image)
        questions, questions_score = sort_questions(questions)
        questions, questions_score = question_nms(questions, questions_score, 0.7)
        self.questions = question_filter(questions, questions_score, 0.2, 5)
        self.nbr_asked_questions = 0
        time.sleep(0.5)

        while self.speak_listen.is_talking():
            time.sleep(0.1)

        self.speak("J'ai "+ str(len(self.questions))+" questions à vous poser")
        self.set_state(STATE_START)

    def question_state(self):

        if self.sub_state == 0:
            self.listen_off()
            self.animator.normal()
            self.speak(generate_question_presentation(self.nbr_asked_questions ))        

            question = self.questions[0]
            self.nbr_asked_questions += 1

            if len(self.questions) > 1:
                self.questions = self.questions[1:]
            else:
                self.questions = []

            self.question_image = question.create_image(self.object_image)
            if self.question_image is None:
                self.get_logger().info("Trying to send None image")
            self.animator.set_custom_img(self.question_image)
            self.animator.show()
            self.speak("Quel est cet objet?")

            self.sub_state = 1
            return
        
        if self.sub_state == 2:
            self.listen_off()
            self.animator.sad()
            self.speak("Je suis désolé de ne pas avoir compris.")
            self.speak("Je vais reposer ma question.")

            if self.question_image is None:
                self.get_logger().info("Trying to send None image")
            self.animator.set_custom_img(self.question_image)
            self.animator.show()
            self.speak("Quel est cet objet?")
            self.sub_state = 1
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            self.animator.remove_custom_img()
            return
        
        if user_input != "":
            text = "An IA ask to the user \"Quel est cet objet?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_answer_choice("What is the object?", text, choices=objects_choices)
            if answer is None:
                self.listen_off()
                self.speak("Désolé, je n'ai pas compris")
            else:
                self.question_answer = answer
                self.animator.remove_custom_img()
                self.set_state(STATE_VALIDATE_QUESTION)

    def validate_question_state(self):
        if self.sub_state == 0:
            self.listen_off()
            self.sub_state = 1
            self.animator.happy()
            self.animator.remove_custom_img()
            self.speak("Selon ma compréhension, l'objet se trouve dans la catégorie "+self.question_answer)
            self.speak("Est-ce que j'ai bien compris?")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            self.animator.remove_custom_img()
            return

        if user_input != "":
            text = "An IA ask to the user \"Selon ma compréhension, l'objet se trouve dans la catégorie "+self.question_answer+". Est-ce que j'ai bien compris?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the object is "+self.question_answer+"?", text, yes_threshold = 0.65)

            if answer:
                self.set_state(STATE_SAVE_ANSWER)
            else:
                self.set_state(STATE_QUESTION)
                self.sub_state = 2

    def save_answer_state(self):
        self.listen_off()
        self.speak("Réponse enregistrée")

        second_phrase_id = random.randint(0,9)
        if second_phrase_id == 0:
            self.speak("Je vous remercie de m'aider dans mon apprentissage")
        if second_phrase_id == 1:
            self.speak("Je me sens maintenant un peut plus intelligent")
        if second_phrase_id == 2:
            self.speak("J'espère que cette information me sera utile")
        if second_phrase_id == 3:
            self.speak("Je rêve qu'un jour, je puisse être aussi intelligent que vous")

        self.set_state(STATE_START)


def main(args=None):

    rclpy.init()

    node = Experiment_node()

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()