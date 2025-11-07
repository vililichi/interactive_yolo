import rclpy
from rclpy.node import Node

from speaklisten import SpeakListenTTOP
from llm_interpreter import LLMInterpreter
from ttop_animator import AnimatorTTOP
from .src.camera import Camera
from .src.model_bbox_clip import Model
from .src.question_sorting import sort_questions, question_filter, question_nms, question_filter_size
from .src.question_presentation import generate_question_presentation
from .src.data import DataManager
from .src.learned_object import LearnedObject

import random

from threading import Thread

import time
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
STATE_INIT = 12

objects_classes = [
    "sandwich",
    "pâtes",
    "salade",
    "pizza",
    "boisson",
    "pain",
    "oeuf",
    "viande",
    "fruit et légume",
    "dessert",
    "objet invalide"
]

objects_choices = objects_classes + [
    "aucune de ces réponses"
]

class Question():
    def __init__(self):
        pass

class Experiment_node(Node):

    def __init__(self):
        super().__init__('experiment_orchestrator_node')

        self.state = STATE_INIT
        self.sub_state = 0
        self.last_state = (STATE_INIT, 0)
        self.transcript = ""

        self.speak_listen = SpeakListenTTOP(self)
        self.speak_listen.talk_end_cb = self.reset_last_speak
        self.speak_listen.voice_detected_end_cb = self.reset_last_speak
        self.speak_listen.voice_processing_end_cb = self.reset_last_speak
        self.llm_interpreter = LLMInterpreter()
        self.animator = AnimatorTTOP(self)
        self.camera = Camera(self, 'interactive_yolo/experiment_image_input', 'interactive_yolo/experiment_image_request')

        self.listen_off()

        self.model = Model(classes_list=objects_classes)

        self.object_image = None

        self.nbr_question = 0
        self.remaining_question : List[Question] = []
        self.last_speak_time = time.time()

        self.nbr_image = 10
        self.image_done_counter = 0

        self.forced_end = False

        # Learning and dataset
        self.data_manager = DataManager()
        self.forced_memory_instance = None



        self.state_machine_thread = Thread(target=self.state_machine_loop, daemon=True)
        self.state_machine_thread.start()

    def add_to_transcript(self, agent:str, text:str):
        transcript_Str = "["+agent+"] "+text
        self.get_logger().info("<TRANSCRIPT>: "+transcript_Str)
        self.transcript += transcript_Str + "\n"

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
            elif self.state == STATE_INIT:
                self.init_state()
            else:
                raise ValueError(f"Invalid state: {self.state}")
            time.sleep(0.05) 

    def set_state(self, state):
        self.last_state = (self.state, self.sub_state)
        self.state = state
        self.sub_state = 0
        self.get_logger().info("Set state: "+str(self.state))

    def time_since_last_speak(self)->float:
        return time.time()-self.last_speak_time

    def return_to_last_state(self):
        self.state, self.sub_state = self.last_state
        self.get_logger().info("Return to last state: "+str(self.state))

    def check_for_experiment_end(self, user_input)->bool:
        
        if user_input == "":
            return False
        
        self.listen_off()
        text = "An IA and an user are doing an experiment where the user try to teach something to the IA."
        text +="\nDuring the dialog, the user says \""+user_input+"\""
        answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user want to end the experiment?", text, yes_threshold = 0.8)
        if answer:
            self.set_state(STATE_CONFIRM_END)
        return answer

    def reset_last_speak(self):
        self.last_speak_time = time.time()

    def listen_on(self):
        if not self.speak_listen.is_listening() and not self.speak_listen.is_talking():
            self.speak_listen.start_listen()
    
    def listen_off(self):

        self.speak_listen.stop_listen()
        while not self.speak_listen.listen_texts.empty():
            self.speak_listen.listen_texts.get()

    def voice_detected(self)->bool:
        return self.speak_listen.is_voice_detected() or self.speak_listen.is_voice_processing_active()

    def try_listen(self):
       listen_result = self.speak_listen.get_listen()
       if len(listen_result)>0:
        self.add_to_transcript("INPUT", listen_result)
       return listen_result
    
    def speak(self, text):
        self.add_to_transcript("OUTPUT", text)
        self.speak_listen.speak(text)
        time.sleep(0.25)
        while self.speak_listen.is_talking():
            time.sleep(0.05)
        return

    def init_state(self):
        ###### Load instance #####
        self.selected_instance = self.select_instance()
        learning_list = self.data_manager.list_learnings(self.selected_instance)
        if "last" in learning_list:
            self.learning = self.data_manager.load_learning(self.selected_instance, "last")
        else:
            self.get_logger().info("No learning found in "+self.selected_instance+", new learning created")
            self.learning = []
        
        ###### add learning to model #####
        self.model.reset_learning()
        for learned_object in self.learning:
            self.model.add_learned_object(learned_object)

        ###### reset image counter #####
        self.image_done_counter = 0
        self.forced_end = False

        ###### Init a variable to remove repeat in saving sentences #####
        self.last_saving_sentence_id = 10

        ###### sleep #####
        self.set_state(STATE_SLEEP)
        self.animator.sleep()

    def sleep_state(self):

        if not self.animator.actual_emotion == "sleep":
            self.animator.sleep()

        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            self.listen_off()
            text = """A scientist talk to an user to explain an experiment with a robot named T-Top.
            Sometime T-Top is also named stoppe or christophe.
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
            
    def create_new_instance(self)->str:

        validation_sets= self.data_manager.list_validation_set()
        learning_sets= self.data_manager.list_learning_set()
        transcript_sets = self.data_manager.list_transcript_set()

        i = 0
        while True:
            instance_id = str(i)
            if instance_id in validation_sets:
                i += 1
                continue
            if instance_id in learning_sets:
                i += 1
                continue
            if instance_id in transcript_sets:
                i += 1
                continue

            self.data_manager.add_validation_set(instance_id)
            self.data_manager.add_learning_set(instance_id)
            self.data_manager.add_transcript_set(instance_id)

            return instance_id

    def select_instance(self)->str:

        if self.forced_memory_instance is not None:
            return str(self.forced_memory_instance)

        return self.create_new_instance()
    
        #instance_size = []
        #for instance in self.memory_instances:
        #    nbr_validation = len(self.data_manager.list_images(instance))
        #    instance_size.append(nbr_validation)

        #min_instance = None
        #min_instance_size = None
        #for i in range(len(instance_size)):
        #    if min_instance_size is None:
        #        min_instance = [self.memory_instances[i],]
        #        min_instance_size = instance_size[i]

        #    elif instance_size[i] == min_instance_size:
        #        min_instance.append(self.memory_instances[i])
            
        #    elif instance_size[i] < min_instance_size:
        #        min_instance = [self.memory_instances[i],]
        #        min_instance_size = instance_size[i]
        
        #if min_instance is None:
        #    raise(Exception("No instance available"))
        
        #if len(min_instance) == 1:
        #    return min_instance[0]
        
        #id = random.randint(0, len(min_instance)-1 )
        #return min_instance[id]


    def confirm_start_state(self):

        if self.sub_state == 0:
            self.listen_off()
            self.animator.happy()
            self.speak("Bonjour! Voulez-vous commencez l'expérience?")
            self.animator.normal()
            self.sub_state = 1
            return
        
        if not self.animator.actual_emotion == "normal":
            self.animator.normal()

        if self.time_since_last_speak() > 30 and not self.voice_detected():
            self.listen_off()
            self.speak("Personne ne semble vouloir me répondre. Je me rendors.")
            self.set_state(STATE_SLEEP)
        
        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            self.listen_off()
            text = "An IA ask to the user \"Voulez-vous commencez l'expérience?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user validate?", text, yes_threshold = 0.5)

            if answer:
                self.set_state(STATE_IMAGE_CAPTURE)
            else:
                self.listen_off()
                self.speak("J'ai compris que vous ne voulez pas commencer l'expérience. Je me rendors.")
                self.set_state(STATE_SLEEP)
        
    def confirm_end_state(self):
        
        if self.sub_state == 0:
            self.listen_off()
            self.animator.sad()
            self.speak("Jai l'impression que vous voulez arrêter l'expérience. Ai-je raison?")
            self.animator.normal()
            self.sub_state = 1
            return
        
        if not self.animator.actual_emotion == "normal":
            self.animator.normal()

        if self.time_since_last_speak() > 15 and not self.voice_detected():
            if self.sub_state == 2:
                self.listen_off()
                self.animator.sad()
                self.speak("Je vais prendre l'abscence de réponse pour un oui.")
                self.animator.normal()
                self.set_state(STATE_END)
                self.forced_end = True
                return
            else:
                self.listen_off()
                self.animator.sad()
                self.speak("J'ai l'impression que vous voulez arrêter l'expérience. Ai-je raison?")
                self.animator.normal()
                self.sub_state = 2
                return
        
        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            self.listen_off()
            text = "An IA ask to the user \"Jai l'impression que vous voulez arrêter l'expérience.Est-ce que j'ai raison?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user want to end the experiment?", text, yes_threshold = 0.5)

            if answer:
                self.set_state(STATE_END)
                self.forced_end = True
            else:
                self.listen_off()
                self.animator.happy()
                self.speak("Alors, continuons l'expérience.")
                self.animator.normal()
                self.return_to_last_state()

    def start_state(self):

        if len(self.questions) > 0:
            self.set_state(STATE_QUESTION)
        else:
            self.listen_off()
            self.animator.happy()
            self.speak("J'ai terminé de poser mes questions sur cette image.")
            self.set_state(STATE_END)
            self.forced_end = False
        
    def end_state(self):

        ##### save data #####

        if self.forced_end:
            self.listen_off()
            self.animator.happy()
            self.speak("Merci pour votre participation à l'expérience.")
            self.animator.sleep()
            self.set_state(STATE_INIT)

        else:
            now = str(int(time.time()))
            image_name = "image_"+now
            learning_name = "learning_"+now
            transcript_name = "letranscript_arning_"+now
            self.data_manager.register_image_in_validation_set(self.selected_instance, image_name, self.object_image)
            self.data_manager.register_transcript_in_transcript_set(self.selected_instance, transcript_name, self.transcript)
            self.data_manager.save_learning(self.selected_instance, learning_name, self.learning)
            self.data_manager.save_learning(self.selected_instance, "last", self.learning)

            self.image_done_counter += 1
            self.get_logger().info("self.image_done_counter: "+str(self.image_done_counter))
            self.get_logger().info("self.nbr_image: "+str(self.nbr_image))
            if self.image_done_counter < self.nbr_image:
                self.set_state(STATE_IMAGE_CAPTURE)
                self.sub_state = 3
            else:
                self.listen_off()
                self.animator.happy()
                self.speak("Merci pour votre participation à l'expérience.")
                self.animator.sleep()
                self.set_state(STATE_INIT)

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
            self.listen_off()
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
            self.speak("Parfait, commençons l'expérience. D'abord, placez votre repas sur le napperon et donnez moi un signal pour que je prenne une photo.")
            self.animator.normal()
            return
        
        if self.time_since_last_speak() > 30 and not self.voice_detected():
            self.listen_off()
            self.speak("Placez votre repas sur le napperon et donnez moi un signal pour que je prenne une photo.")
            return

        
        if self.sub_state == 2:
            self.listen_off()
            self.sub_state = 1
            self.animator.normal()
            self.speak("Recommençons la photo. Placez votre repas sur le napperon et donnez moi un signal.")
            return
        
        if self.sub_state == 3:
            self.listen_off()
            self.sub_state = 1
            self.animator.normal()
            self.speak("Passons au prochain repas. Placez votre repas sur le napperon et donnez moi un signal.")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            return
        
        if user_input != "":
            self.listen_off()
            text = """A user is doing an experiment with a robot named T-Top.
            T-Top is waiting for a signal to take a picture.
            T-Top said \"Placez votre repas sur le centre de la table et donnez moi un signal pour que je prenne une photo.\""""
            text +="\nThe user said \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do T-Top received the signal to take a picture?", text, yes_threshold=0.7)
            if answer:
                self.animator.check_table()
                self.speak("Cheeeeeeeeeeese")
                self.object_image = self.camera.capture()
                #self.speak("Image capturée.")

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
            self.speak("La photo est-elle adéquate?")
            return
        
        if self.time_since_last_speak() > 10 and not self.voice_detected():
            self.listen_off()
            self.animator.set_custom_img(self.object_image)
            self.speak("La photo est-elle adéquate?")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if user_input != "":
            self.listen_off()
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
        self.animator.thinking()
        text = "J'analyse la photo."
        self.speak_listen.speak(text)
        self.add_to_transcript("OUTPUT", text)
   
        ###### question generation #####
        questions = self.model.generate_question(self.object_image)
        questions, questions_score = sort_questions(questions)
        questions, questions_score = question_filter_size(questions, questions_score, max_relative_size=0.85)
        questions, questions_score = question_nms(questions, questions_score, 0.7)
        self.questions = question_filter(questions, questions_score, 0.15, 3)
        self.nbr_asked_questions = 0
        time.sleep(0.2)

        while self.speak_listen.is_talking():
            time.sleep(0.05)

        self.speak("J'ai "+ str(len(self.questions))+" questions à vous poser")
        self.set_state(STATE_START)

    def question_state(self):

        if self.sub_state == 0:
            self.listen_off()
            self.animator.normal()
            self.speak(generate_question_presentation(self.nbr_asked_questions ))        

            self.question = self.questions[0]
            self.nbr_asked_questions += 1

            if len(self.questions) > 1:
                self.questions = self.questions[1:]
            else:
                self.questions = []

            self.question_image = self.question.create_image(self.object_image)
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
            self.speak("Je vais reposer la question.")

            if self.question_image is None:
                self.get_logger().info("Trying to send None image")
            self.animator.set_custom_img(self.question_image)
            self.animator.show()
            self.speak("Quel est cet objet?")
            self.sub_state = 1
            return
        
        if self.time_since_last_speak() > 30 and not self.voice_detected():
            self.listen_off()
            self.animator.set_custom_img(self.question_image)
            self.speak("Quel est cet objet?")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            self.animator.remove_custom_img()
            return
        
        if user_input != "":
            self.listen_off()
            text = "An IA ask to the user \"Quel est cet objet?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_answer_choice("What is the object?", text, choices=objects_choices)
            if answer is None or answer == "aucune de ces réponses":
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
            self.speak("L'objet est dans la catégorie "+self.question_answer+". Ai-je bien compris?")
            return
        
        if self.time_since_last_speak() > 30 and not self.voice_detected():
            self.listen_off()
            self.animator.remove_custom_img()
            self.speak("L'objet est dans la catégorie "+self.question_answer+". Ai-je bien compris?")
            return
        
        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            self.animator.remove_custom_img()
            return

        if user_input != "":
            self.listen_off()
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
        self.learning.append(LearnedObject(self.question_answer, self.question.embedding, self.question.mask_conf))
        self.speak("Réponse enregistrée")


        second_phrase_id = self.last_saving_sentence_id
        while(second_phrase_id == self.last_saving_sentence_id):
            second_phrase_id = random.randint(0,50)
        self.last_saving_sentence_id = second_phrase_id

        if second_phrase_id == 0:
            self.speak("Je vous remercie de m'aider dans mon apprentissage")
        if second_phrase_id == 1:
            self.speak("Je me sens maintenant un peut plus intelligent")
        if second_phrase_id == 2:
            self.speak("J'espère que cette information me sera utile")

        self.set_state(STATE_START)

def main(args=None):

    rclpy.init()

    node = Experiment_node()

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()