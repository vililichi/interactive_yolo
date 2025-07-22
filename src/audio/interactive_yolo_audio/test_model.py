import speech_recognition as sr
import json
import shutil
import os
import time

from urllib.request import urlretrieve
import zipfile

from interactive_yolo_utils import workspace_dir

input_list = [
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_1.wav"), "T-Top, j'aimerais commencer l'expérience."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_2.wav"), "Oui, je veux commencer l'expérience."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_3.wav"), "C'est fait."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_4.wav"), "Oui, la photo est adéquate."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_5.wav"), "Je suis prête."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_6.wav"), "Oui, la photo est adéquate."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_7.wav"), "Prends la photo."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_8.wav"), "Oui, la photo est adéquate."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_9.wav"), "Un papier, c'est un papier."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_10.wav"), "Oui."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_11.wav"), "C'est un papier."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_12.wav"), "Oui, tu as bien compris, T-Top."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_13.wav"), "Humain. C'est un humain."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_14.wav"), "Non, tu n'as pas bien compris."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_15.wav"), "C'est un humain."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_16.wav"), "Oui, tu as bien compris."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_17.wav"), "C'est un humain."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_18.wav"), "Non, tu n'as pas bien compris."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_19.wav"), "Cet objet est un humain."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_20.wav"), "Oui, tu as bien compris."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_21.wav"), "C'est une table."),
    (os.path.join(workspace_dir(), "audio_test_sample", "slice_22.wav"), "Oui, tu as bien compris.")
]

def pre_calculate_wer(reference:str, hypothesis:str):
    formated_reference = reference.lower().replace(",", "").replace(".", "").replace("!","").replace("?","").replace("  "," ").strip(" ")
    formated_hypothesis = hypothesis.lower().replace(",", "").replace(".", "").replace("!","").replace("?","").replace("  "," ").strip(" ")
    
    ref_words = formated_reference.split()
    hyp_words = formated_hypothesis.split()
    # Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    # Total number of words in the reference text
    total_words = len(ref_words)
    # Calculating the Word Error Rate (WER)
    return (substitutions + deletions + insertions), total_words

def calculate_wer(error:int, total_words:int):
    wer = error / total_words
    return wer

VOSK_MODEL_PATH = os.path.join(workspace_dir(),"models","vosk")

def install_vosk_model():
    model_url= "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip"
    model_zip = "vosk-model-fr-0.22.zip"
    model_unzip_dir = "temporary_vosk_install"
    model_to_move = os.path.join(model_unzip_dir,"vosk-model-fr-0.22")

    # check if model dir exist
    if not os.path.isdir(VOSK_MODEL_PATH):

        #download model
        print("downloading vosk model")
        urlretrieve(model_url, model_zip)

        #unzip model
        print("unzip vosk model")
        with zipfile.ZipFile(model_zip,"r") as zip_ref:
            zip_ref.extractall(model_unzip_dir)

        print("install vosk model")
        shutil.move(model_to_move, VOSK_MODEL_PATH)

        print("cleaning")
        os.removedirs(model_unzip_dir)
        os.remove(model_zip)

        print("installation finished")

whisper_size = "large"
test_google = False
test_whisper = True
test_fast_whisper = False
test_vosk = False

r = sr.Recognizer()

install_vosk_model()


if( test_vosk ):
    from vosk import Model
    r.vosk_model = Model(VOSK_MODEL_PATH)

google_errors = 0
google_total_words = 0
google_time = 0

whisper_errors = 0
whisper_total_words = 0
whisper_time = 0

fast_whisper_errors = 0
fast_whisper_total_words = 0
fast_whisper_time = 0

vosk_errors = 0
vosk_total_words = 0
vosk_time = 0

for file, ref_text in input_list:
    audio_source = sr.AudioFile(file)
    with audio_source as source:

        audio = r.record(source)

        google_result = ""
        whisper_result = ""
        fast_whisper_result = ""
        vosk_result = ""

        if test_google:
            start_time = time.time()
            try:
                google_result = r.recognize_google(audio, language='fr-FR')
            except:
                pass
            google_time += time.time() - start_time

        if test_whisper:
            start_time = time.time()
            try:
                whisper_result = r.recognize_whisper(audio, language='french', model=whisper_size)
            except:
                pass
            whisper_time += time.time() - start_time

        if test_fast_whisper:
            start_time = time.time()
            try:
                fast_whisper_result = r.recognize_faster_whisper(audio, language='fr', model=whisper_size)
            except:
                pass
            fast_whisper_time += time.time() - start_time


        if test_vosk:
            start_time = time.time()
            try:
                vosk_result = json.loads(r.recognize_vosk(audio, language='fr-FR')).get('text', '')
            except:
                pass
            vosk_time += time.time() - start_time

        if test_google:
            errors, total_words = pre_calculate_wer(ref_text, google_result)
            google_errors += errors
            google_total_words += total_words

        if test_whisper:
            errors, total_words = pre_calculate_wer(ref_text, whisper_result)
            whisper_errors += errors
            whisper_total_words += total_words

        if test_fast_whisper:
            errors, total_words = pre_calculate_wer(ref_text, fast_whisper_result)
            fast_whisper_errors += errors
            fast_whisper_total_words += total_words

        if test_vosk:
            errors, total_words = pre_calculate_wer(ref_text, vosk_result)
            vosk_errors += errors
            vosk_total_words += total_words


        print("_______________________________________________________________________")
        print("**Reference**")
        print(ref_text)

        if test_google:
            print("**Google**")
            print(google_result)

        if test_whisper:
            print("**Whisper**")
            print(whisper_result)

        if test_fast_whisper:
            print("**Fast Whisper**")
            print(fast_whisper_result)

        if test_vosk:
            print("**Vosk**")
            print(vosk_result)

        print("_______________________________________________________________________")

print("_______________________________________________________________________")
if test_google:
    print("**Google**")
    print("WER: ",calculate_wer(google_errors, google_total_words)*100,"%")
    print("time: ",google_time,"s")

if test_whisper:
    print("**Whisper**")
    print("WER: ",calculate_wer(whisper_errors, whisper_total_words)*100,"%")
    print("time: ",whisper_time,"s")

if test_fast_whisper:
    print("**Fast Whisper**")
    print("WER: ",calculate_wer(fast_whisper_errors, fast_whisper_total_words)*100,"%")
    print("time: ",fast_whisper_time,"s")

if test_vosk:
    print("**Vosk**")
    print("WER: ",calculate_wer(vosk_errors, vosk_total_words)*100,"%")
    print("time: ",vosk_time,"s")