from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

from speaklisten import SpeakListenTTOP
from llm_interpreter import LLMInterpreter
from ttop_animator import AnimatorTTOP

from threading import Thread

import time

STATE_SLEEP = 0
STATE_CONFIRM_START = 1
STATE_START = 2
STATE_CONFIRM_END = 3
STATE_END = 4


class Experiment_node(Node):

    def __init__(self):
        super().__init__('experiment_orchestrator_node')

        self.state = STATE_SLEEP
        self.sub_state = 0
        self.last_state = (STATE_SLEEP, 0)

        self.speak_listen = SpeakListenTTOP(self)
        self.llm_interpreter = LLMInterpreter()
        self.animator = AnimatorTTOP(self)

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
            else:
                raise ValueError(f"Invalid state: {self.state}")
            time.sleep(0.1) 

    def set_state(self, state):
        self.last_state = (self.state, self.sub_state)
        self.state = state
        self.sub_state = 0

    def return_to_last_state(self):
        self.state, self.sub_state = self.last_state

    def listen_on(self):
        if not self.speak_listen.is_listening() and not self.speak_listen.is_talking():
            self.speak_listen.start_listen()
    
    def listen_off(self):
        if self.speak_listen.is_listening():
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

        self.listen_on()
        user_input = self.try_listen()

        if self.sub_state == 0:
            self.animator.sleep()
            self.sub_state = 1
            return

        if user_input != "":
            text = """A scientist talk to an user to explain an experiment with a robot named T-Top.
            T-Top is waiting for a signal to start the experiment.
            The signal always containt the word the experiment."""
            text +="\nThe robot hear \""+user_input+"\""
            text += """
            The robot hesitate to start the experiment because it does not know if the message is for him.
            The majority of the messages are not for him.
            """
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do T-Top received the signal to start the experiment?", text, yes_threshold=0.8)
            if answer:
                self.set_state(STATE_CONFIRM_START)
    
    def confirm_start_state(self):

        self.listen_on()
        user_input = self.speak_listen.get_listen()

        if self.sub_state == 0:
            self.animator.happy()
            self.speak("Bonjour!")
            self.animator.normal()
            self.speak("Voulez-vous commencez l'expérience?")
            self.sub_state = 1
            return

        if user_input != "":
            text = "An IA ask to the user \"Voulez-vous commencez l'expérience?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user validate?", text, yes_threshold = 0.5)

            if answer:
                self.set_state(STATE_START)
            else:
                self.speak("J'ai compris que bous ne voulez pas commencer l'expérience.")
                self.speak("Je vais donc me rendormir.")
                self.set_state(STATE_SLEEP)
        
    def confirm_experiment_end_state(self):
        
        self.listen_on()
        user_input = self.speak_listen.get_listen()

        if self.sub_state == 0:
            self.animator.sad()
            self.speak("Jai l'impression que vous voulez arrêter l'expérience.")
            self.animator.normal()
            self.speak("Est-ce que j'ai raison?")
            self.sub_state = 1
            return

        if user_input != "":
            text = "An IA ask to the user \"Jai l'impression que vous voulez arrêter l'expérience.Est-ce que j'ai raison?\""
            text +="\nThe user answer with \""+user_input+"\""
            answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user want to end the experiment?", text, yes_threshold = 0.5)

            if answer:
                self.set_state(STATE_END)
            else:
                self.animator.happy()
                self.speak("Alors, nous allons continuer l'expérience.")
                self.animator.normal()
                self.return_to_last_state()

    def start_State(self):

        self.listen_on()
        user_input = self.try_listen()

        if self.check_for_experiment_end(user_input):
            return
        
    def end_state(self):
        self.animator.happy()
        self.speak("Merci pour votre participation à l'expérience.")
        self.set_state(STATE_SLEEP)

    def check_for_experiment_end(self, user_input)->bool:
        text = "An IA and an user are doing an experiment where the user try to teach something to the IA."
        text +="\nDuring the dialog, the user says \""+user_input+"\""
        answer = self.llm_interpreter.ask_question_asymetric_yes_no("Do the user want to end the experiment?", text, yes_threshold = 0.8)
        if answer:
            self.set_state(STATE_CONFIRM_END)
        return answer
    


def main(args=None):

    rclpy.init()

    node = Experiment_node()

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()