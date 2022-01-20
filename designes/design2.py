# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog

from predict import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 500)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(15)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 501, 91))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.btn_keyPoints_on_picture = QtWidgets.QPushButton(self.centralwidget)
        self.btn_keyPoints_on_picture.setGeometry(QtCore.QRect(60, 350, 40, 40))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.btn_keyPoints_on_picture.setFont(font)
        self.btn_keyPoints_on_picture.setCheckable(False)
        self.btn_keyPoints_on_picture.setObjectName("btn_keyPoints_on_picture")
        self.btn_keyPoints_in_video = QtWidgets.QPushButton(self.centralwidget)
        self.btn_keyPoints_in_video.setGeometry(QtCore.QRect(60, 410, 40, 40))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.btn_keyPoints_in_video.setFont(font)
        self.btn_keyPoints_in_video.setCheckable(False)
        self.btn_keyPoints_in_video.setObjectName("btn_keyPoints_on_picture_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(130, 350, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(130, 410, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(60, 160, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(15)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.comboBox_nn_model = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_nn_model.setGeometry(QtCore.QRect(60, 210, 271, 31))
        self.comboBox_nn_model.setObjectName("comboBox_nn_model")
        self.comboBox_nn_model.addItem("")
        self.comboBox_nn_model.addItem("")
        self.comboBox_nn_model.addItem("")
        self.btn_more_info = QtWidgets.QPushButton(self.centralwidget)
        self.btn_more_info.setGeometry(QtCore.QRect(340, 210, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.btn_more_info.setFont(font)
        self.btn_more_info.setCheckable(False)
        self.btn_more_info.setObjectName("btn_more_info")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 100, 381, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(25)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(60, 280, 381, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(25)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.add_connectors()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "KeyPoints Detector"))
        self.label.setText(_translate("MainWindow", "Hello, User"))
        self.btn_keyPoints_on_picture.setText(_translate("MainWindow", "1"))
        self.btn_keyPoints_in_video.setText(_translate("MainWindow", "2"))
        self.label_2.setText(_translate("MainWindow", "Detect KeyPoints On Picture"))
        self.label_3.setText(_translate("MainWindow", "Detect KeyPoints In Video"))
        self.label_4.setText(_translate("MainWindow", "Using NN Model"))
        self.comboBox_nn_model.setItemText(0, _translate("MainWindow", "The Fastest"))
        self.comboBox_nn_model.setItemText(1, _translate("MainWindow", "The most accurate"))
        self.comboBox_nn_model.setItemText(2, _translate("MainWindow", "The balanced one"))
        self.btn_more_info.setText(_translate("MainWindow", "more info"))
        self.label_5.setText(_translate("MainWindow", "Select NN Model"))
        self.label_6.setText(_translate("MainWindow", "Select what you want to do"))

    def add_connectors(self):
        self.btn_keyPoints_in_video.clicked.connect(self.browse_video_files)
        self.btn_keyPoints_on_picture.clicked.connect(self.browse_photo_files)

    def browse_photo_files(self):
        file_name = QFileDialog.getOpenFileName(self.centralwidget, 'Open file', '', 'All Files (*)')
        # self.filename.setText(file_name[0])
        if file_name[0] != '':
            predict(file_name[0])

    def browse_video_files(self):
        file_name = QFileDialog.getOpenFileName(self.centralwidget, 'Open file', '', 'All Files (*)')
        # self.filename.setText(file_name[0])


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())