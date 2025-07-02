from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class QuestionWithImage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self._valid_callback = None
        self._cancel_callback = None

        self._question_label = QtWidgets.QLabel("Question")
        self._question_label.setAlignment(Qt.AlignCenter)
        self._image = ScaledImage()
        self._image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._answer_label = QtWidgets.QLineEdit("Réponse")
        self._answer_label.setAlignment(Qt.AlignCenter)
        
        self._button_valid = QtWidgets.QPushButton("Valider")
        self._button_cancel = QtWidgets.QPushButton("Annuler")
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.addWidget(self._button_valid)
        self._button_layout.addWidget(self._button_cancel)

        self._button_valid.clicked.connect(self.validButtonClicked)
        self._button_cancel.clicked.connect(self.cancelButtonClicked)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._question_label)
        self._layout.addWidget(self._image)
        self._layout.addWidget(self._answer_label)
        self._layout.addLayout(self._button_layout)

        self.setLayout(self._layout)
    
    def setQuestion(self, question:str):
        self._question_label.setText(question)

    def setAnswer(self, answer:str):
        self._answer_label.setText(answer)

    def setValidButtonLabel(self, label:str):
        self._button_valid.setText(label)

    def setCancelButtonLabel(self, label:str):
        self._button_cancel.setText(label)

    def setImage(self, cv_image:np.ndarray):
        self._image.setImage(cv_image)

    def validButtonClicked(self):
        if self._valid_callback is not None:
            self._valid_callback(self._answer_label.text())
    
    def cancelButtonClicked(self):
        if self._cancel_callback is not None:
            self._cancel_callback()
    
    def setValidCallback(self, callback):
        self._valid_callback = callback
    
    def setCancelCallback(self, callback):
        self._cancel_callback = callback

class CaptureWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self._capture_callback = None

        self.cv_image_1 = None
        self.cv_image_2 = None
        self.use_img_2 = False

        self.swap_button_name = "Swap"
        self.mode_1_name = "Image 1"
        self.mode_2_name = "Image 2"

        self._image = ScaledImage()
        self._image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        self._button_capture = QtWidgets.QPushButton("Capture")
        self._button_swap = QtWidgets.QPushButton(self.swap_button_name+" : "+self.mode_1_name)
        self._button_layout = QtWidgets.QVBoxLayout()
        self._button_layout.addWidget(self._button_swap)
        self._button_layout.addWidget(self._button_capture)

        self._button_capture.clicked.connect(self.captureButtonClicked)
        self._button_swap.clicked.connect(self.swapButtonClicked)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._image)
        self._layout.addLayout(self._button_layout)

        self.setLayout(self._layout)

    def __set_image(self):
        if(self.use_img_2):
            if(self.cv_image_2 is not None):
                self._image.setImage(self.cv_image_2)
        else:
            if(self.cv_image_1 is not None):
                self._image.setImage(self.cv_image_1)

    def __set_swap_button_label(self):
        label = self.swap_button_name+" : "
        if(self.use_img_2):
            label += self.mode_2_name
        else:
            label += self.mode_1_name

        self._button_swap.setText(label)


    def setCaptureButtonLabel(self, label:str):
        self._button_capture.setText(label)

    def setSwapButtonLabel(self, label:str):
        self.swap_button_name = label
        self.__set_swap_button_label()
    
    def setSwapButtonMode1Name(self, label:str):
        self.mode_1_name = label
        self.__set_swap_button_label()

    def setSwapButtonMode2Name(self, label:str):
        self.mode_2_name = label
        self.__set_swap_button_label()

    def setImage1(self, cv_image:np.ndarray):

        self.cv_image_1 = cv_image
        self.__set_image()

    def setImage2(self, cv_image:np.ndarray):

        self.cv_image_2 = cv_image
        self.__set_image()

        

    def captureButtonClicked(self):
        if self._capture_callback is not None:
            self._capture_callback(self.cv_image_1, self.cv_image_2, self.use_img_2)

    def swapButtonClicked(self):
        self.use_img_2 = not self.use_img_2

        self.__set_image()
        self.__set_swap_button_label()
    
    def setCaptureCallback(self, callback):
        self._capture_callback = callback

class ScaledImage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self._cv_image = None
        self._qpixmap = None
        self._image_label = QtWidgets.QLabel()

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._image_label)
        self.setLayout(self._layout)

        self._img_size = 0.9

    def _convert_cv_qt(self, cv_img, size:float=0.75):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(int(self.width()*size), int(self.height()*size), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def _setImageWithSize(self, cv_image:np.ndarray, size:float=0.75):
        self._cv_image = cv_image
        self._qpixmap = self._convert_cv_qt(cv_image, size)
        self._image_label.setPixmap(self._qpixmap)

    def setImage(self, cv_image:np.ndarray):
        self._setImageWithSize(cv_image, self._img_size)

    def resizeEvent(self, event):
        out = super().resizeEvent(event)
        if self._cv_image is not None:
            self._setImageWithSize(self._cv_image, self._img_size)

        return out