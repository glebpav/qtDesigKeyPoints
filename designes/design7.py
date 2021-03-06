# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design7.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(486, 647)
        MainWindow.setStyleSheet("#centralWidget{\n"
"background-color: qlineargradient(spread:\n"
"pad, x1:0, y1:1, x2:0, y2:0, \n"
"stop:0 rgba(0, 0, 0, 255), \n"
"stop:0.05 rgba(14, 8, 73, 255), \n"
"stop:0.36 rgba(28, 17, 145, 255), \n"
"stop:0.6 rgba(126, 14, 81, 255), \n"
"stop:0.75 rgba(234, 11, 11, 255), \n"
"stop:0.79 rgba(244, 70, 5, 255), \n"
"stop:0.86 rgba(255, 136, 0, 255), \n"
"stop:0.935 rgba(239, 236, 55, 255));\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("#centralwidget{\n"
"background-color: qlineargradient(spread:\n"
"pad, x1:0, y1:1, x2:0, y2:0, \n"
"stop:0 #febcca, \n"
"stop:0.935 #a3b2fd);\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(20, 20, 20, 20)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel")
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setMaximumSize(QtCore.QSize(1000, 16777215))
        font = QtGui.QFont()
        font.setFamily("Corbel")
        font.setPointSize(25)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(40, -1, 40, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboBox_nn_model_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_nn_model_2.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        self.comboBox_nn_model_2.setFont(font)
        self.comboBox_nn_model_2.setStyleSheet("QComboBox  {\n"
"    background-color: rgb(252, 229, 113);\n"
"    border-radius: 10px;\n"
"}")
        self.comboBox_nn_model_2.setObjectName("comboBox_nn_model_2")
        self.comboBox_nn_model_2.addItem("")
        self.comboBox_nn_model_2.addItem("")
        self.comboBox_nn_model_2.addItem("")
        self.horizontalLayout.addWidget(self.comboBox_nn_model_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(40, -1, 40, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.load_custom = QtWidgets.QPushButton(self.centralwidget)
        self.load_custom.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.load_custom.setFont(font)
        self.load_custom.setStyleSheet("QPushButton{\n"
"    background-color: rgb(    252, 229, 113);\n"
"    border: none;\n"
"     color: rgb(0, 0, 0);\n"
"    border-radius: 8px\n"
"}\n"
"QPushButton:hover { \n"
"    background-color: rgb(235, 214, 105); \n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(195, 177, 87); \n"
"}\n"
"")
        self.load_custom.setCheckable(False)
        self.load_custom.setObjectName("load_custom")
        self.horizontalLayout_2.addWidget(self.load_custom)
        self.btn_more_info = QtWidgets.QPushButton(self.centralwidget)
        self.btn_more_info.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.btn_more_info.setFont(font)
        self.btn_more_info.setStyleSheet("QPushButton{\n"
"    background-color: rgb(    252, 229, 113);\n"
"    border: none;\n"
"     color: rgb(0, 0, 0);\n"
"    border-radius: 8px\n"
"}\n"
"QPushButton:hover { \n"
"    background-color: rgb(235, 214, 105); \n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(195, 177, 87); \n"
"}\n"
"")
        self.btn_more_info.setCheckable(False)
        self.btn_more_info.setObjectName("btn_more_info")
        self.horizontalLayout_2.addWidget(self.btn_more_info)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel")
        font.setPointSize(25)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.checkBox_show_p_names = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        self.checkBox_show_p_names.setFont(font)
        self.checkBox_show_p_names.setStyleSheet("QCheckBox::indicator {\n"
"    width: 15px;\n"
"    height: 15px;\n"
"    background-color: rgb(252, 229, 113);\n"
"    border-radius: 7px;\n"
" }\n"
"QCheckBox::indicator:unchecked  {\n"
"    background-color: rgb(191, 174, 85);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked   {\n"
"    background-color: rgb(252, 229, 113);\n"
"}")
        self.checkBox_show_p_names.setObjectName("checkBox_show_p_names")
        self.horizontalLayout_3.addWidget(self.checkBox_show_p_names)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.checkBox_show_connections = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        self.checkBox_show_connections.setFont(font)
        self.checkBox_show_connections.setStyleSheet("QCheckBox::indicator {\n"
"    width: 15px;\n"
"    height: 15px;\n"
"    background-color: rgb(252, 229, 113);\n"
"    border-radius: 7px;\n"
" }\n"
"QCheckBox::indicator:unchecked  {\n"
"    background-color: rgb(191, 174, 85);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked   {\n"
"    background-color: rgb(252, 229, 113);\n"
"}")
        self.checkBox_show_connections.setObjectName("checkBox_show_connections")
        self.horizontalLayout_4.addWidget(self.checkBox_show_connections)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout.addItem(spacerItem3)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel")
        font.setPointSize(25)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.btn_keyPoints_on_picture_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_keyPoints_on_picture_2.setMaximumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.btn_keyPoints_on_picture_2.setFont(font)
        self.btn_keyPoints_on_picture_2.setStyleSheet("QPushButton{\n"
"    background-color: rgb(    252, 229, 113);\n"
"    border: none;\n"
"     color: rgb(0, 0, 0);\n"
"    border-radius: 8px\n"
"}\n"
"QPushButton:hover { \n"
"    background-color: rgb(235, 214, 105); \n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(195, 177, 87); \n"
"}\n"
"")
        self.btn_keyPoints_on_picture_2.setCheckable(False)
        self.btn_keyPoints_on_picture_2.setObjectName("btn_keyPoints_on_picture_2")
        self.horizontalLayout_5.addWidget(self.btn_keyPoints_on_picture_2)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem5)
        self.btn_keyPoints_in_video_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_keyPoints_in_video_2.setMaximumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.btn_keyPoints_in_video_2.setFont(font)
        self.btn_keyPoints_in_video_2.setStyleSheet("QPushButton{\n"
"    background-color: rgb(    252, 229, 113);\n"
"    border: none;\n"
"     color: rgb(0, 0, 0);\n"
"    border-radius: 8px\n"
"}\n"
"QPushButton:hover { \n"
"    background-color: rgb(235, 214, 105); \n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(195, 177, 87); \n"
"}\n"
"")
        self.btn_keyPoints_in_video_2.setCheckable(False)
        self.btn_keyPoints_in_video_2.setObjectName("btn_keyPoints_in_video_2")
        self.horizontalLayout_6.addWidget(self.btn_keyPoints_in_video_2)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Corbel Light")
        font.setPointSize(15)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_6.addWidget(self.label_9)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout.addItem(spacerItem6)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Hello, User"))
        self.label_5.setText(_translate("MainWindow", "Select NN Model"))
        self.comboBox_nn_model_2.setItemText(0, _translate("MainWindow", "The Fastest"))
        self.comboBox_nn_model_2.setItemText(1, _translate("MainWindow", "The most accurate"))
        self.comboBox_nn_model_2.setItemText(2, _translate("MainWindow", "The balanced one"))
        self.load_custom.setText(_translate("MainWindow", "custom"))
        self.btn_more_info.setText(_translate("MainWindow", "more info"))
        self.label_7.setText(_translate("MainWindow", "Select necessary options"))
        self.checkBox_show_p_names.setText(_translate("MainWindow", "Show names of points"))
        self.checkBox_show_connections.setText(_translate("MainWindow", "Show connections"))
        self.label_6.setText(_translate("MainWindow", "Select what you want to do"))
        self.btn_keyPoints_on_picture_2.setText(_translate("MainWindow", "1"))
        self.label_4.setText(_translate("MainWindow", "Detect KeyPoints On Picture"))
        self.btn_keyPoints_in_video_2.setText(_translate("MainWindow", "2"))
        self.label_9.setText(_translate("MainWindow", "Detect KeyPoints In Video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
