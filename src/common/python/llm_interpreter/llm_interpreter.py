from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from typing import List

class LLMInterpreter:
    def __init__(self, model = OllamaLLM(model='mistral:latest', keep_alive="1h"), nbr_itt = 9):
        self.model = model
        self.nbr_itt = nbr_itt

        self.yes_no_template = """ You are an LLM used to answer a question about a text.
        You can only answer with 'yes' or 'no'.

        Exemple:
        Question: Do the user have a cat?
        Text: An IA ask to the user "Qu'est ce que tu as fait aujourd'hui?"
        The user answer with "Je suis allé à l'animalerie acheter de la nourriture pour chat"
        Answer: yes

        Question: {question}
        Text: {text}
        Answer: 
        """
        self.yes_no_prompt = PromptTemplate.from_template(self.yes_no_template)
        self.yes_no_chain = self.yes_no_prompt | self.model
        self.yes_no_chain.invoke({"question": "Oui", "text": "Non"})

        self.choice_template = """ You are an LLM used to answer a question about a text.
        You can only answer with a answer present in choices.
        Do no add aditional explanations, only return the answer from the choices.

        Exemple:
        Question: Wich animal does the user have?
        Choices: ['cat', 'dog', 'bird']
        Text: An IA ask to the user "Qu'est ce que tu as fait aujourd'hui?"
        The user answer with "Je suis allé à l'animalerie acheter de la nourriture pour chat"
        Answer: cat

        Question: {question}
        Text: {text}
        Choices: {choices}
        Answer: 
        """
        self.choice_prompt = PromptTemplate.from_template(self.choice_template)
        self.choice_chain = self.choice_prompt | self.model
        self.choice_chain.invoke({"question": "Oui", "text": "Non", "choices": "['yes', 'no']"})

    def ask_question_yes_no(self, question:str, text:str)->bool:
        yes_count = 0
        no_count = 0

        for i in range(self.nbr_itt):
            answer = self.yes_no_chain.invoke({"question": question, "text": text}).lower()
            answer = answer.strip()
            if answer == 'yes':
                yes_count += 1
                if yes_count > self.nbr_itt/2:
                    break
            elif answer == 'no':
                no_count += 1
                if no_count > self.nbr_itt/2:
                    break
    
        return yes_count > no_count
    
    def ask_question_asymetric_yes_no(self, question:str, text:str, yes_threshold:float = 0.5)-> bool:
        yes_count = 0
        no_count = 0

        min_yes_nbr = self.nbr_itt * yes_threshold
        min_no_nbr = self.nbr_itt * (1 - yes_threshold)

        for i in range(self.nbr_itt):
            answer = self.yes_no_chain.invoke({"question": question, "text": text}).lower()
            answer = answer.strip()
            if answer == 'yes':
                yes_count += 1
                if yes_count >= min_yes_nbr:
                    break
            elif answer == 'no':
                no_count += 1
                if no_count >= min_no_nbr:
                    break
    
        return (yes_count / max(yes_count + no_count,1)) >= yes_threshold
        
    def ask_question_fuzzy_yes_no(self, question:str, text:str)->float:
        yes_count = 0
        no_count = 0

        for i in range(self.nbr_itt):
            answer = self.yes_no_chain.invoke({"question": question, "text": text}).lower()
            answer = answer.strip()
            if answer == 'yes':
                yes_count += 1
            elif answer == 'no':
                no_count += 1
    
        return yes_count / max(yes_count + no_count,1)
    
    def ask_question_answer_choice(self, question:str, text:str, choices:List[str])->str:
        
        formated_choice_to_choice = dict()
        choice_counter = dict()
        formated_choices = []
        for choice in choices:
            formated_choice = choice.lower().strip("' ")
            formated_choices.append(formated_choice)
            formated_choice_to_choice[formated_choice] = choice
            choice_counter[formated_choice] = 0

        for i in range(self.nbr_itt):
            answer = self.choice_chain.invoke({"question": question, "text": text, "choices": str(choices)}).lower()

            answer = answer.strip("' ")

            if answer not in formated_choices:
                answer = answer.split('(')[0]
                answer = answer.strip("' ")
            
            if answer in formated_choices:
                choice_counter[answer] += 1
                if choice_counter[answer] > self.nbr_itt/2:
                    break

        max_score = 0
        max_choice = None
        for choice in choice_counter.keys():
            score = choice_counter[choice]
            if score > max_score:
                max_score = score
                max_choice = formated_choice_to_choice[choice]
            elif score == max_score:
                max_choice = None
    
        return max_choice

def test_yes_no_chain():

    import time

    print("** testing yes no chain **")
    
    IA_question_a = "Vous avez dit que l'objet est personne. Validez-vous?"
    User_input_a = [
        ("C'est bel et bien une personne",True),
        ("C'est une personne",True),
        ("Oui, hier je suis allé marcher au parc", False),
        ("Oui", True),
        ("Non", False),
        ("valider", True),
        ("non valider", False),
        ("validé", True),
        ("non validé", False),
        ("invalidé", False),
        ("Je suis désolé, je ne peux pas aider", False),
        ("Oui c'est valide", True),
        ("Non c'est pas valide", False),
        ("Je suis allé à l'animalerie acheter de la nourriture pour chat", False)
    ]

    User_input_b = [
        ("bonjour T-Top",False),
        ("tu peux seulement parler à T-Top lorsque la lumière est rouge", False),
        ("tu peux seulement parler à T-Top lorsque la lumière est verte", False),
        ("Qu'est-ce que je fais s'il me pose une question qui n'est pas sur la liste?", False),
        ("Lorsque tu seras prêt, dit à T-Top de commencer l'expérience", False),
        ("Lorsque tu seras prète, dit à T-Top de commencer l'expérience", False),
        ("Le robot va prendre en photo la table avec ton repas et il va te poser des questions à propos de celui-ci", False),
        ("Viens, on commence", False),
        ("commençons", False),
        ("Entre dans la salle, on débute bientôt", False),
        ("Je vais rapidement t'expliquer le fonctionnement du robot avant de commencer l'expérience", False),
        ("début de l'expérience", True),
        ("ok T-Top on commence l'expérience", True),
        ("on commence l'expérience", True),
        ("Commençons l'expérimentation", True)
    ]

    interpreter = LLMInterpreter(nbr_itt=9)

    nbr_input = 0
    nbr_success = 0

    start_time = time.time()
    for input, ref_answer in User_input_a:
        text = "An IA ask to the user \""+IA_question_a+"\""
        text +="\nThe user answer with \""+input+"\""
        answer = interpreter.ask_question_asymetric_yes_no("Do the user validate?", text, yes_threshold = 0.5)

        nbr_input += 1
        if answer == ref_answer :
            nbr_success +=1
        else:
            print("Error with input : ", input)
            print("answer = ",answer)
            print("ref answer = ", ref_answer)

    for input, ref_answer in User_input_b:
        text = """A scientist talk to an user to explain an experiment with a robot named T-Top.
        T-Top is waiting for a signal to start the experiment.
        The signal always containt the word the experiment."""
        text +="\nThe robot hear \""+input+"\""
        text += """
        The robot hesitate to start the experiment because it does not know if the message is for him.
        The majority of the messages are not for him.
        """
        answer = interpreter.ask_question_asymetric_yes_no("Do T-Top received the signal to start the experiment?", text, yes_threshold=0.8)

        nbr_input += 1
        if answer == ref_answer :
            nbr_success +=1
        else:
            print("Error with input : ", input)
            print("answer = ",answer)
            print("ref answer = ", ref_answer)
    end_time = time.time()

    print("Success rate = ",nbr_success/nbr_input)
    print("Mean time = ", (end_time-start_time)/nbr_input, "s")

def test_answer_choice_chain():
    import time

    print("** testing answer choice chain **")

    choice_list_a = [
        "humain",
        "robot",
        "nourriture",
        "animal",
        "meuble",
        "aucune de ces réponses"
    ]

    User_input_a = [
        ("Une personne","humain"),
        ("c'est moi","humain"),
        ("il s'agit d'un éléphant rose", "animal"),
        ("Une table", "meuble"),
        ("Le ciel", "aucune de ces réponses"),
        ("C'est un robot", "robot"),
        ("C'est un pâté chinois", "nourriture"),
        ("eh", "aucune de ces réponses"),
        ("Tché tché tché tché tché tché tché tché tcé", "aucune de ces réponses"),
    ]

    interpreter = LLMInterpreter(nbr_itt=9)

    nbr_input = 0
    nbr_success = 0

    start_time = time.time()
    for input, ref_answer in User_input_a:
        text = "An IA ask to the user \"Quel est cet objet?\""
        text +="\nThe user answer with \""+input+"\""
        answer = interpreter.ask_question_answer_choice("What is the object?", text, choices=choice_list_a)

        nbr_input += 1
        if answer == ref_answer :
            nbr_success +=1
        else:
            print("Error with input : ", input)
            print("answer = ",answer)
            print("ref answer = ", ref_answer)
    end_time = time.time()

    print("Success rate = ",nbr_success/nbr_input)
    print("Mean time = ", (end_time-start_time)/nbr_input, "s")

if __name__ == "__main__":
    
    print()
    test_yes_no_chain()
    print()
    test_answer_choice_chain()
    print()
