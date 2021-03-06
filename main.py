import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from playsound import playsound

from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from predictor import Darknet
from utils import count_nomask_violations, sound_signal, check_if_violates_any, TimeForSoundChecker, count_noshields_violations


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    changeNumNoMasks = pyqtSignal(str)

    changeNumNoShields = pyqtSignal(str)

    def run(self):
        checker = TimeForSoundChecker()

        # from camera
        cap = cv2.VideoCapture(0)
        # from video
        # cap = cv2.VideoCapture("crowd.mp4")

        # https://www.tutorialexample.com/python-pyqt5-play-wav-file-a-completed-guide-pyqt-tutorial/
        url = QtCore.QUrl.fromLocalFile('./sound_assets/alarm_one.wav')
        content = QtMultimedia.QMediaContent(url)
        player = QtMultimedia.QMediaPlayer()
        player.setMedia(content)
        player.setVolume(50.0)

        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                image, detections = darknet.predict(frame)
                num_no_masks = count_nomask_violations(detections)
                num_no_shields = count_noshields_violations(detections)
                if checker.has_been_a_second():
                    if check_if_violates_any(detections):
                        player.play()
                ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1024, 768)
                self.changePixmap.emit(Pic)
                self.changeNumNoMasks.emit(num_no_masks)
                self.changeNumNoShields.emit(num_no_shields)


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

    @pyqtSlot(str)
    def setNumNomasks(self, num_no_masks):
        self.stat_one_value.setText(num_no_masks)

    @pyqtSlot(str)
    def setNumNoshields(self, num_no_shields):
        self.stat_two_value.setText(num_no_shields)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1400, 850)
        # create a label
        self.label = QLabel(self)
        self.label.move(30, 30)
        self.label.resize(1280, 720)

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
        self.stat_one.setText("No face masks:")
        self.stat_one.move(1100, 70)
        self.stat_one.resize(200, 50)

        # Value one
        label_font = QtGui.QFont()
        label_font.setPointSize(25)
        self.stat_one_value = QtWidgets.QLabel(self)
        self.stat_one_value.setFont(label_font)
        self.stat_one_value.setObjectName("stat_one_value")
        self.stat_one_value.setText("")
        self.stat_one_value.move(1150, 120)
        self.stat_one_value.resize(200, 50)

        # Label Two
        label_font = QtGui.QFont()
        label_font.setPointSize(20)
        self.stat_two = QtWidgets.QLabel(self)
        self.stat_two.setFont(label_font)
        self.stat_two.setObjectName("stat_two")
        self.stat_two.setText("No face shields:")
        self.stat_two.move(1100, 200)
        self.stat_two.resize(200, 50)

        # Value one
        label_font = QtGui.QFont()
        label_font.setPointSize(25)
        self.stat_two_value = QtWidgets.QLabel(self)
        self.stat_two_value.setFont(label_font)
        self.stat_two_value.setObjectName("stat_two_value")
        self.stat_two_value.setText("")
        self.stat_two_value.move(1150, 250)
        self.stat_two_value.resize(200, 50)

        # Connection to threads
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changeNumNoMasks.connect(self.setNumNomasks)
        th.changeNumNoShields.connect(self.setNumNoshields)
        th.start()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    darknet = Darknet(1280, 720)
    ex = App()
    sys.exit(app.exec_())
