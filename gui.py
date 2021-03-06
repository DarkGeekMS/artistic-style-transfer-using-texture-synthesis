# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import sys,os
from PyQt5.QtWidgets import  QTableWidget,QTableWidgetItem, QFileDialog
from PyQt5.QtGui import QPixmap
from src.style_transfer import style_transfer

class Ui_ArtisticStyleTransfer(object):
    def setupUi(self, ArtisticStyleTransfer):
        ArtisticStyleTransfer.setObjectName("ArtisticStyleTransfer")
        ArtisticStyleTransfer.resize(1246, 1004)
        ArtisticStyleTransfer.setStyleSheet("background-color: lightcyan")
        font = QtGui.QFont()
        font.setFamily("FreeSerif")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        ArtisticStyleTransfer.setFont(font)
        ArtisticStyleTransfer.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.centralwidget = QtWidgets.QWidget(ArtisticStyleTransfer)
        self.centralwidget.setObjectName("centralwidget")
        self.Stylization = QtWidgets.QPushButton(self.centralwidget)
        self.Stylization.setGeometry(QtCore.QRect(850, 840, 131, 61))
        self.Stylization.setObjectName("Stylization")
        self.Stylization.setStyleSheet("background-color: lightgray")
        self.content = QtWidgets.QPushButton(self.centralwidget)
        self.content.setGeometry(QtCore.QRect(360, 850, 161, 51))
        self.content.setObjectName("content")
        self.style = QtWidgets.QPushButton(self.centralwidget)
        self.style.setGeometry(QtCore.QRect(560, 850, 161, 51))
        self.style.setObjectName("style")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(340, 90, 881, 701))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.images = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.images.setContentsMargins(1, 1, 1, 1)
        self.images.setObjectName("images")
        self.segImg = QtWidgets.QLabel(self.gridLayoutWidget)
        self.segImg.setObjectName("segImg")
        self.images.addWidget(self.segImg, 1, 1, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.contentImg = QtWidgets.QLabel(self.gridLayoutWidget)
        self.contentImg.setObjectName("contentImg")
        self.images.addWidget(self.contentImg, 0, 1, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.styleImg = QtWidgets.QLabel(self.gridLayoutWidget)
        self.styleImg.setObjectName("styleImg")
        self.images.addWidget(self.styleImg, 1, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.stylizedImg = QtWidgets.QLabel(self.gridLayoutWidget)
        self.stylizedImg.setObjectName("stylizedImg")
        self.images.addWidget(self.stylizedImg, 0, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.sigmaS = QtWidgets.QSpinBox(self.centralwidget)
        self.sigmaS.setGeometry(QtCore.QRect(200, 100, 101, 31))
        self.sigmaS.setObjectName("sigmaS")
        self.sigmaS.setRange(1,100)
        self.sigmaS.setStyleSheet("background-color: white")
        self.sigmaR = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.sigmaR.setGeometry(QtCore.QRect(200, 150, 101, 31))
        self.sigmaR.setObjectName("sigmaR")
        self.sigmaR.setDecimals(2)
        self.sigmaR.setRange(0,1)
        self.sigmaR.setStyleSheet("background-color: white")
        self.denoisingItr = QtWidgets.QSpinBox(self.centralwidget)
        self.denoisingItr.setGeometry(QtCore.QRect(200, 200, 101, 31))
        self.denoisingItr.setObjectName("denoisingItr")
        self.denoisingItr.setRange(1,3)
        self.denoisingItr.setStyleSheet("background-color: white")
        self.learningItr = QtWidgets.QSpinBox(self.centralwidget)
        self.learningItr.setGeometry(QtCore.QRect(200, 430, 101, 31))
        self.learningItr.setObjectName("learningItr")
        self.learningItr.setRange(1,10)
        self.learningItr.setStyleSheet("background-color: white")
        self.irlsItr = QtWidgets.QSpinBox(self.centralwidget)
        self.irlsItr.setGeometry(QtCore.QRect(200, 310, 101, 31))
        self.irlsItr.setObjectName("irlsItr")
        self.irlsItr.setRange(1,10)
        self.irlsItr.setStyleSheet("background-color: white")
        self.DSS = QtWidgets.QLabel(self.centralwidget)
        self.DSS.setGeometry(QtCore.QRect(10, 100, 151, 31))
        self.DSS.setObjectName("DSS")
        self.DSR = QtWidgets.QLabel(self.centralwidget)
        self.DSR.setGeometry(QtCore.QRect(10, 150, 151, 31))
        self.DSR.setObjectName("DSR")
        self.RSV = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.RSV.setGeometry(QtCore.QRect(200, 370, 101, 31))
        self.RSV.setObjectName("RSV")
        self.RSV.setDecimals(2)
        self.RSV.setRange(0.7,0.9)
        self.RSV.setStyleSheet("background-color: white")
        self.DOP = QtWidgets.QSpinBox(self.centralwidget)
        self.DOP.setGeometry(QtCore.QRect(200, 490, 101, 31))
        self.DOP.setObjectName("DOP")
        self.DOP.setRange(1,4)
        self.DOP.setStyleSheet("background-color: white")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 660, 101, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        font = QtGui.QFont()
        font.setFamily("FreeSerif")
        font.setPointSize(17)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 200, 171, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 260, 241, 41))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(False)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 310, 151, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(0, 370, 191, 31))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(30, 430, 151, 31))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(30, 490, 151, 31))
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(30, 560, 161, 31))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(20, 620, 241, 21))
        font = QtGui.QFont()
        font.setFamily("FreeSerif")
        font.setPointSize(17)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.imgSize = QtWidgets.QSpinBox(self.centralwidget)
        self.imgSize.setGeometry(QtCore.QRect(200, 660, 101, 31))
        self.imgSize.setObjectName("imgSize")
        self.imgSize.setRange(400,600)
        self.imgSize.setStyleSheet("background-color: white")
        self.SegmentationTechnique = QtWidgets.QComboBox(self.centralwidget)
        self.SegmentationTechnique.setGeometry(QtCore.QRect(40, 735, 241, 41))
        self.SegmentationTechnique.setObjectName("SegmentationTechnique")
        self.SegmentationTechnique.setStyleSheet("selection-color: black")
        self.SegmentationTechnique.addItem("")
        self.SegmentationTechnique.setItemText(0, "   Segmentation Technique")
        self.SegmentationTechnique.addItem("")
        self.SegmentationTechnique.addItem("")
        self.SegmentationTechnique.addItem("")
        self.colorTransfer = QtWidgets.QComboBox(self.centralwidget)
        self.colorTransfer.setGeometry(QtCore.QRect(40, 820, 241, 41))
        self.colorTransfer.setObjectName("colorTransfer")
        self.colorTransfer.setStyleSheet("selection-color: black")
        self.colorTransfer.addItem("")
        self.colorTransfer.setItemText(0, "  ColorTransfer Technique")
        self.colorTransfer.addItem("")
        self.colorTransfer.addItem("")
        self.PatchNum = QtWidgets.QComboBox(self.centralwidget)
        self.PatchNum.setGeometry(QtCore.QRect(205, 560, 91, 31))
        self.PatchNum.setObjectName("PatchNum")
        self.PatchNum.setStyleSheet("selection-color: black")
        self.PatchNum.addItem("")
        self.PatchNum.addItem("")
        self.PatchNum.addItem("")
        self.PatchNum.addItem("")
        self.PatchNum.addItem("")
        ArtisticStyleTransfer.setCentralWidget(self.centralwidget)

        self.content_path = None
        self.style_path = None

        self.retranslateUi(ArtisticStyleTransfer)
        QtCore.QMetaObject.connectSlotsByName(ArtisticStyleTransfer)

    def retranslateUi(self, ArtisticStyleTransfer):
        _translate = QtCore.QCoreApplication.translate
        ArtisticStyleTransfer.setWindowTitle(_translate("ArtisticStyleTransfer", "Artistic Style Transfer"))
        ArtisticStyleTransfer.setWhatsThis(_translate("ArtisticStyleTransfer", "<html><head/><body><p><br/></p></body></html>"))
        self.Stylization.setText(_translate("ArtisticStyleTransfer", "Apply Style"))
        self.content.setText(_translate("ArtisticStyleTransfer", "Choose Content"))
        self.style.setText(_translate("ArtisticStyleTransfer", "Choose Style"))
        self.segImg.setText(_translate("ArtisticStyleTransfer", "Segmentation Mask"))
        self.contentImg.setText(_translate("ArtisticStyleTransfer", "Content Image"))
        self.styleImg.setText(_translate("ArtisticStyleTransfer", "Style Image"))
        self.stylizedImg.setText(_translate("ArtisticStyleTransfer", "Stylized Image"))
        self.DSS.setText(_translate("ArtisticStyleTransfer", "sigma_s denoising"))
        self.DSR.setText(_translate("ArtisticStyleTransfer", "sigma_r denoising"))
        self.label_3.setText(_translate("ArtisticStyleTransfer", "Image Size"))
        self.label_4.setText(_translate("ArtisticStyleTransfer", "Denoising Parameters :"))
        self.label_5.setText(_translate("ArtisticStyleTransfer", "Number of Iterations"))
        self.label_6.setText(_translate("ArtisticStyleTransfer", "Learning Parameters :"))
        self.label_7.setText(_translate("ArtisticStyleTransfer", "IRLs No. Iterations"))
        self.label_8.setText(_translate("ArtisticStyleTransfer", "Robust Statistical Value"))
        self.label_9.setText(_translate("ArtisticStyleTransfer", "Learning Iterations"))
        self.label_10.setText(_translate("ArtisticStyleTransfer", "Depth of Pyramid"))
        self.label_12.setText(_translate("ArtisticStyleTransfer", "    Patch Numbers"))
        self.label_13.setText(_translate("ArtisticStyleTransfer", "General Parameters :"))
        self.SegmentationTechnique.setItemText(1, _translate("ArtisticStyleTransfer", "Blurred Edges"))
        self.SegmentationTechnique.setItemText(2, _translate("ArtisticStyleTransfer", "Convex Hull"))
        self.SegmentationTechnique.setItemText(3, _translate("ArtisticStyleTransfer", "morphological_chan_vese"))
        self.colorTransfer.setItemText(1, _translate("ArtisticStyleTransfer", "Histogram Matching"))
        self.colorTransfer.setItemText(2, _translate("ArtisticStyleTransfer", "LAB Color Space"))
        self.PatchNum.setItemText(0, _translate("ArtisticStyleTransfer", "5"))
        self.PatchNum.setItemText(1, _translate("ArtisticStyleTransfer", "4"))
        self.PatchNum.setItemText(2, _translate("ArtisticStyleTransfer", "3"))
        self.PatchNum.setItemText(3, _translate("ArtisticStyleTransfer", "2"))
        self.PatchNum.setItemText(4, _translate("ArtisticStyleTransfer", "1"))


        self.Stylization.clicked.connect(self.applyStyle)
        self.content.clicked.connect(self.getContent)
        self.style.clicked.connect(self.getStyle)

    def getContent(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)

        if dlg.exec_():
            self.content_path = dlg.selectedFiles()[0]




    def getStyle(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            self.style_path = dlg.selectedFiles()[0]



    def applyStyle(self):

        if (self.content_path == None):
            self.content_path = 'data/content/eagles.jpg'

        if (self.style_path == None):
            self.style_path = 'data/style/van_gogh.jpg'  

        if (self.PatchNum.currentIndex() == 0):
            patch_sizes = [33,21,13,9,5]
            sub_gaps = [28,18,8,5,3]
        elif (self.PatchNum.currentIndex() == 1):
            patch_sizes = [33,21,13,9]
            sub_gaps = [28,18,8,5]    
        elif (self.PatchNum.currentIndex() == 2):
            patch_sizes = [33,21,13]
            sub_gaps = [28,18,8]    
        elif (self.PatchNum.currentIndex() == 3):
            patch_sizes = [33,21]
            sub_gaps = [28,18]    
        else:
            patch_sizes = [33]
            sub_gaps = [28]    

        if (self.SegmentationTechnique.currentIndex() == 1):
            seg_mode = 2
        elif (self.SegmentationTechnique.currentIndex() == 2):
            seg_mode = 1
        else:
            seg_mode = 0       

        if (self.colorTransfer.currentIndex() == 1):
            ct_mode = '1'
        else:
            ct_mode = '0'      

        content, style, seg_mask, X = style_transfer(self.content_path, self.style_path, self.imgSize.value(), self.DOP.value(), patch_sizes, sub_gaps, self.irlsItr.value(), \
        self.learningItr.value(), self.RSV.value(), 5.0, seg_mode, ct_mode, self.sigmaS.value(), self.sigmaR.value(), self.denoisingItr.value())

        pixmap = QtGui.QPixmap(self.content_path)
        self.contentImg.setPixmap(pixmap)
        
        pixmap = QtGui.QPixmap(self.style_path)
        self.styleImg.setPixmap(pixmap)

        pixmap = QtGui.QPixmap("outputs/output.png")
        self.stylizedImg.setPixmap(pixmap)

        pixmap = QtGui.QPixmap("outputs/seg_output.png")
        self.segImg.setPixmap(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_ArtisticStyleTransfer()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
