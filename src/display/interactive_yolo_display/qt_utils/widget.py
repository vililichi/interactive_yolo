from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QObject, pyqtSignal
import cv2
import numpy as np

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

        self._img_size = 0.85

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
    
class MySignalEmitter(QObject):
    # Define a custom signal with a value
    signal = pyqtSignal(str)

class ConsoleWidget(QtWidgets.QWidget):
    def __init__(self): 
        super().__init__()

        self.out_text = ""
        self.input_callback = None

        self.log_signal = MySignalEmitter()
 
        # create objects
        label = QtWidgets.QLabel()
        self.le = QtWidgets.QLineEdit()
        self.te = QtWidgets.QTextEdit()
        self.log_signal.signal.connect(self.te.setPlainText)

        # layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.le)
        layout.addWidget(self.te)
        self.setLayout(layout) 

        # create connection
        self.le.returnPressed.connect(self._enter_cb)
    
    def log(self, role:str, message:str):
        self.out_text += "["+role+"]: "+message+"\n"
        self.log_signal.signal.emit(self.out_text)
    
    def _log(self):
        self.out_text += "["+self.role+"]: "+self.message+"\n"
        self.te.setPlainText(self.out_text)

    def _enter_cb(self):
        text = self.le.text()
        if self.input_callback is not None:
            self.input_callback(text)