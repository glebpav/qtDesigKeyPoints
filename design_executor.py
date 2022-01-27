import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from designes.design7 import Ui_MainWindow
from dialog_more_info import Ui_Dialog_more_info
from predict import *


class Window_executor(QtWidgets.QMainWindow):
    def __init__(self):
        super(Window_executor, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Points Detector")
        super(Window_executor, self)
        self.add_connectors()

    def add_connectors(self):
        self.ui.btn_keyPoints_in_video_2.clicked.connect(self.browse_video_files)
        self.ui.btn_keyPoints_on_picture_2.clicked.connect(self.browse_photo_files)
        self.ui.btn_more_info.clicked.connect(self.open_dialog_more_info)

    def browse_photo_files(self):
        file_name = QFileDialog.getOpenFileName(self.ui.centralwidget, 'Open file', '', 'All Files (*)')
        # self.filename.setText(file_name[0])

        print(file_name)
        show_names_of_points = self.ui.checkBox_show_p_names.checkState()
        show_connection = self.ui.checkBox_show_connections.checkState()

        if file_name[0] != '':
            predict(file_name[0], show_connections=show_connection, show_points_names=show_names_of_points)

    def open_dialog_more_info(self):
        self.window = QtWidgets.QDialog()
        self.ui = Ui_Dialog_more_info()
        self.ui.setupUi(self.window)
        self.window.show()

    def browse_video_files(self):
        file_name = QFileDialog.getOpenFileName(self.ui.centralwidget, 'Open file', '', 'All Files (*)')

        show_names_of_points = self.ui.checkBox_show_p_names.checkState()
        show_connection = self.ui.checkBox_show_connections.checkState()

        if file_name[0] != '':
            video_predict(file_name[0], show_connections=show_connection, show_points_names=show_names_of_points)


app = QtWidgets.QApplication([])
application = Window_executor()
application.show()

sys.exit(app.exec())
