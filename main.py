import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from PyQt5 import QtCore, QtGui, QtWidgets
from predictor import Darknet

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                print(darknet.predict(frame))
                # https://stackoverflow.com/a/55468544/6622587
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
                self.changePixmap.emit(Pic)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()


    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1280, 850)
        # create a label
        self.label = QLabel(self)
        self.label.move(30, 30)
        self.label.resize(1280, 720)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

        # Button One
        self.button_one = QtWidgets.QPushButton(self)
        self.button_one.setObjectName("button_one")
        self.button_one.setText("Push Button")
        self.button_one.move(250, 770)
        self.button_one.resize(200, 60)

        # Button Two
        self.button_two = QtWidgets.QPushButton(self)
        self.button_two.setObjectName("button_two")
        self.button_two.setText("Push Button")
        self.button_two.move(550, 770)
        self.button_two.resize(200, 60)

        # Label One
        label_font = QtGui.QFont()
        label_font.setPointSize(20)
        self.stat_one = QtWidgets.QLabel(self)
        self.stat_one.setFont(label_font)
        self.stat_one.setObjectName("stat_one")
        self.stat_one.setText("<stat one>:")
        self.stat_one.move(1020, 70)
        self.stat_one.resize(200, 50)

        # Value one
        label_font = QtGui.QFont()
        label_font.setPointSize(25)
        self.stat_one_value = QtWidgets.QLabel(self)
        self.stat_one_value.setFont(label_font)
        self.stat_one_value.setObjectName("stat_one_value")
        self.stat_one_value.setText("*value*")
        self.stat_one_value.move(1070, 120)
        self.stat_one_value.resize(200, 50)

        # Label Two
        label_font = QtGui.QFont()
        label_font.setPointSize(20)
        self.stat_two = QtWidgets.QLabel(self)
        self.stat_two.setFont(label_font)
        self.stat_two.setObjectName("stat_two")
        self.stat_two.setText("<stat two>:")
        self.stat_two.move(1020, 200)
        self.stat_two.resize(200, 50)

        # Value one
        label_font = QtGui.QFont()
        label_font.setPointSize(25)
        self.stat_two_value = QtWidgets.QLabel(self)
        self.stat_two_value.setFont(label_font)
        self.stat_two_value.setObjectName("stat_two_value")
        self.stat_two_value.setText("*value*")
        self.stat_two_value.move(1070, 250)
        self.stat_two_value.resize(200, 50)



        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    darknet = Darknet(1280, 720)
    ex = App()
    sys.exit(app.exec_())