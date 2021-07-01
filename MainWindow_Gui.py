# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow_Gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 579)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setMinimumSize(QtCore.QSize(640, 480))
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.horizontalLayout_2.addWidget(self.imgLabel)
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.AlgoNameComboBox = QtWidgets.QComboBox(self.groupBox_7)
        self.AlgoNameComboBox.setToolTip("")
        self.AlgoNameComboBox.setObjectName("AlgoNameComboBox")
        self.verticalLayout_2.addWidget(self.AlgoNameComboBox)
        self.verticalLayout.addWidget(self.groupBox_7)
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.AccuracyButton = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AccuracyButton.sizePolicy().hasHeightForWidth())
        self.AccuracyButton.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/accuracy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.AccuracyButton.setIcon(icon)
        self.AccuracyButton.setObjectName("AccuracyButton")
        self.gridLayout_6.addWidget(self.AccuracyButton, 2, 0, 1, 1)
        self.TrainButton = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TrainButton.sizePolicy().hasHeightForWidth())
        self.TrainButton.setSizePolicy(sizePolicy)
        self.TrainButton.setToolTip("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/train.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.TrainButton.setIcon(icon1)
        self.TrainButton.setObjectName("TrainButton")
        self.gridLayout_6.addWidget(self.TrainButton, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.Accuracy_label = QtWidgets.QLabel(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Accuracy_label.sizePolicy().hasHeightForWidth())
        self.Accuracy_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Accuracy_label.setFont(font)
        self.Accuracy_label.setFrameShape(QtWidgets.QFrame.Box)
        self.Accuracy_label.setText("")
        self.Accuracy_label.setObjectName("Accuracy_label")
        self.verticalLayout.addWidget(self.Accuracy_label)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.BrowseButton = QtWidgets.QPushButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BrowseButton.sizePolicy().hasHeightForWidth())
        self.BrowseButton.setSizePolicy(sizePolicy)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/image files.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.BrowseButton.setIcon(icon2)
        self.BrowseButton.setObjectName("BrowseButton")
        self.gridLayout_7.addWidget(self.BrowseButton, 0, 0, 1, 1)
        self.PredictionButton = QtWidgets.QPushButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PredictionButton.sizePolicy().hasHeightForWidth())
        self.PredictionButton.setSizePolicy(sizePolicy)
        self.PredictionButton.setToolTip("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("images/attendance.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.PredictionButton.setIcon(icon3)
        self.PredictionButton.setObjectName("PredictionButton")
        self.gridLayout_7.addWidget(self.PredictionButton, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.ExitButton = QtWidgets.QPushButton(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ExitButton.sizePolicy().hasHeightForWidth())
        self.ExitButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.ExitButton.setFont(font)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ExitButton.setIcon(icon4)
        self.ExitButton.setObjectName("ExitButton")
        self.verticalLayout.addWidget(self.ExitButton)
        self.gridLayout_4.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Algorithm Selection"))
        self.groupBox.setTitle(_translate("MainWindow", "Training Process"))
        self.AccuracyButton.setToolTip(_translate("MainWindow", "Calculate the accuracy of Dataset"))
        self.AccuracyButton.setText(_translate("MainWindow", "Accuracy"))
        self.TrainButton.setText(_translate("MainWindow", "Training"))
        self.label_2.setText(_translate("MainWindow", "Accuracy (%) :"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Image Recognition"))
        self.BrowseButton.setToolTip(_translate("MainWindow", "Browse Local Image for Recognition"))
        self.BrowseButton.setText(_translate("MainWindow", "Browse Image"))
        self.PredictionButton.setText(_translate("MainWindow", "Prediction"))
        self.ExitButton.setToolTip(_translate("MainWindow", "EXIT Program"))
        self.ExitButton.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
