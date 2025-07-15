from .question import Question
from typing import List, Tuple


def sort_questions(questions:List[Question])->Tuple[List[Question], List[float]]:

    questions_with_score = []
    for question in questions:
        questions_with_score.append((question, compute_question_score(question)))
    
    questions_with_score.sort(key=lambda item: item[1], reverse=True)

    question_out = []
    score_out = []
    for question, score in questions_with_score:
        question_out.append(question)
        score_out.append(score)
    
    return (question_out, score_out)

def compute_question_score(question:Question)->float:
    return (1.0 - question.explain_score) * question.mask_conf

def question_filter(questions:List[Question], scores:List[float], min_score = None, max_number = None):
    
    if min_score is None and max_number is None:
        return questions
    
    if min_score is None:
        min_score = 0

    if max_number is None:
        max_number = len(questions)

    out = []
    nbr_select = 0
    for i in range(len(questions)):
        if scores[i] >= min_score:
            out.append(questions[i])
            nbr_select += 1

        if nbr_select >= max_number:
            break
    
    return out
