from .src.model_bbox_clip import Model
from .src.question_sorting import sort_questions, question_filter, question_nms
from .src.data import DataManager
from matplotlib import pyplot as plt
import cv2

def main():

    ################# CHANGE VALUES HERE #########################
    validation_set = "instance_test"
    validation_img = "image_1759931050"

    learning_set = "instance_test"
    learning_name = "learning_1759931050"

    data_manager = DataManager()

    ################ GENERATE MODEL ####################################
    print("GENERATE MODEL")
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
        "autre accompagnement",
        "objet invalide"
    ]

    model = Model(classes_list=objects_classes)

    learning = []
    if learning_set in data_manager.list_learning_set():
        if learning_name in data_manager.list_learnings(learning_set):
            learning = data_manager.load_learning(learning_set, learning_name)

    for learned_object in learning:
        model.add_learned_object(learned_object)

    ################ LOAD IMAGE ####################################
    print("LOAD IMAGE")
    image = None
    if validation_set in data_manager.list_validation_set():
        if validation_img in data_manager.list_images(validation_set):
            image = data_manager.load_image(validation_set, validation_img)
    
    if image is None:
        print("Image is None")
        return
    
    #################### Generate questions ########################
    print("GANERATE QUESTIONS")
    questions = model.generate_question(image)
    questions, scores = sort_questions(questions)
    #questions, scores = question_nms(questions, scores, 0.9)

    #################### Show all questions ########################
    print("SHOW ALL QUESTIONS")
    for i in range(len(questions)):
        question = questions[i]
        score = scores[i]
        mask_conf = question.mask_conf
        explain_conf = question.explain_score
        centering_score = question.get_centering_score()
        size_score = question.get_size_score()

        print("--------------------")
        print("score = ", score)
        print("mask_conf = ", mask_conf)
        print("explain_conf = ", explain_conf)
        print("centering_score = ", centering_score)
        print("size_score = ", size_score)
        print("--------------------")

        question_img = question.create_image(image)
        question_img = cv2.cvtColor(question_img, cv2.COLOR_BGR2RGB)
        plt.imshow(question_img)
        plt.show()

if __name__ == '__main__':
    main()