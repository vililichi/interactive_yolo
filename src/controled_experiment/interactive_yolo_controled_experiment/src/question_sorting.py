from .question import Question
from typing import List, Tuple
import numpy as np


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
    return (1.0 - question.explain_score) * (question.mask_conf) * (question.get_centering_score()**0.5) * (question.get_size_score()**0.5)

def question_nms(questions:List[Question], scores:List[float], map_threshold = 0.9)->Tuple[List[Question], List[float]]:
    questions_out:List[Question] = []
    scores_out = []

    for i in range(len(questions)):
        validated = True
        for validated_question in questions_out:
            intersection = np.count_nonzero(np.logical_and(validated_question.mask, questions[i].mask))
            union = np.count_nonzero(np.logical_or(validated_question.mask, questions[i].mask))
            if intersection/union > map_threshold:
                validated = False
                break

        if validated:
            questions_out.append(questions[i])
            scores_out.append(scores[i])

    return (questions_out, scores_out)

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
