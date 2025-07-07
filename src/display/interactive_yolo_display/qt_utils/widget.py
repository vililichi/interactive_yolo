from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
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