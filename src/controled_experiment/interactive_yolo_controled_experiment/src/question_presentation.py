import random

count_adj_list =[
    "première",
    "deuxième",
    "troisième",
    "quatrième",
    "cinquième",
    "sixième",
    "septième",
    "huitième",
    "neuvième",
    "dixième",
    "onzième",
    "douzième"
]

def get_count_adj(asked_questions : int):
    if asked_questions < 10:
        return count_adj_list[asked_questions]
    else:
        return "prochaine"

def generate_question_presentation(asked_questions : int):

    rand_number = random.randint(0,4)
    count_adj = get_count_adj(asked_questions)
    presentation = ""

    if rand_number == 0:
        presentation = "Je vais vous poser la "+count_adj+" question."
    elif rand_number == 1:
        presentation = "C'est au tour de la "+count_adj+" question."
    elif rand_number == 2:
        presentation = "Voici la "+count_adj+" question."
    elif rand_number == 3:
        presentation = "La "+count_adj+" question est la suivante."
    elif rand_number == 4:
        presentation = "Passons à la "+count_adj+" question"

    return presentation