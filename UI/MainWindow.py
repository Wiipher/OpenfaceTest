# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import os
import os.path
import threading
import time
from multiprocessing import Process, Queue

import cv2
import numpy as np

from PyQt5.QtCore import pyqtSlot,  QModelIndex
from PyQt5.QtWidgets import QMainWindow,  QFileDialog,  QMessageBox, QInputDialog
from PyQt5.QtGui import QPixmap

from UI.Ui_mainwindow import Ui_MainWindow

from FaceCompare.FaceCompare import FaceCompare

import qimage2ndarray as q2n

camQueue = Queue(256)
camProcQueue = Queue(256)

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        self.openfilename = ''
        self.libdir = ''
        
        self.isSearch = False
        showLibThread = threading.Thread(target = self.showLib)
        showLibThread.setDaemon(True)
        showLibThread.start()
    
    @pyqtSlot(QModelIndex)
    def on_listLib_clicked(self, index):
        libpicPath = self.listLib.item(index.row()).text()
        self.showImg(self.lblLibpic, libpicPath)
    
    @pyqtSlot()
    def on_btnAdd_clicked(self):
        if self.isShow is False:
            img = cv2.imread(self.openfilename)
        elif self.capture.isOpened():
            self.isShow = False
            ret, img = self.capture.read()
        else:
            self.txtInfo.setText("Please connect a camera or choose a photo !") 
            return -1

        img_ad = self.brightAdjust(img)
        if img_ad is not False:
            img = img_ad
        imgRep, point_lt, point_rb  = self.fc.getImgRep(img)
        if imgRep is not False:
            imgtemp = img.copy()
            self.drawCross(imgtemp, point_lt, 25, 2, True)
            self.drawCross(imgtemp, point_rb, 25, 2, False)
            self.showMat(self.lblCamera, imgtemp)             
            if not (self.listLib.count() > 0):
                if not os.path.exists("./Library"):
                    os.makedirs("./Library")
                self.libdir = "./Library"
                addname, ok = QInputDialog.getText(self, "Input Dialog", "Please enter your name :")
                if ok:
                    cv2.imwrite("./Library/{}.jpg".format(addname), img)
                    self.fc.saveRepToCsv("./Library/{}.csv".format(addname), imgRep)
            else:
                addname, ok = QInputDialog.getText(self, "Input Dialog", "Please enter your name :")
                if ok:
                    cv2.imwrite(self.libdir + "/{}.jpg".format(addname), img)  
                    self.fc.saveRepToCsv(self.libdir + "/{}.csv".format(addname), imgRep)

            self.refreshLib()                        
        else:
            self.txtInfo.setText("Unable to find a face !") 
            
        self.isShow = True

    def brightAdjust(self, img):
        img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        imgRep, point_lt, point_rb  = self.fc.getImgRep(img, True)
        if imgRep is False:
            return False
        face_v = img_v[point_lt[0]:point_rb[0], point_lt[1]:point_rb[1]]
        face_mean = cv2.mean(face_v)[0]
        if face_mean > 0 and face_mean < 120:
            ratio = 120 / face_mean
        else:
            ratio = 1
        img_v = np.uint8(np.clip(img_v * ratio, 0, 255))
        dstimg = cv2.cvtColor(cv2.merge([img_h, img_s, img_v]), cv2.COLOR_HSV2BGR)
        return dstimg
    
    @pyqtSlot()
    def on_btnDelete_clicked(self):
        if self.listLib.count() > 0:
            delfile = self.listLib.item(self.listLib.currentRow()).text()
            csvPath = os.path.dirname(delfile)
            (shortname, extension) = os.path.splitext(os.path.basename(delfile))
            csvfile = csvPath + '/' + shortname + '.csv'
            if os.path.exists(delfile):
                os.remove(delfile)
            if os.path.exists(csvfile):
                os.remove(csvfile)
            self.listLib.takeItem(self.listLib.currentRow())
            self.showImg(self.lblLibpic, self.listLib.item(self.listLib.currentRow()).text())
        else:
            self.txtInfo.setText("Please choose a library !") 
    
    @pyqtSlot()
    def on_btnVerify_clicked(self):
        if self.isShow is False:
            img = cv2.imread(self.openfilename)
        elif self.capture.isOpened():
            self.isShow = False
            ret, img = self.capture.read()
        else:
            self.txtInfo.setText("Please connect a camera or choose a photo !") 
            return -1

        img_ad = self.brightAdjust(img)
        if img_ad is not False:
            img = img_ad
        self.compareLib(img)

        self.isShow = True
            
    def compareLib(self, img):
        tempDis = 100
        tempPath = ""
        if len(self.libdir):
            imgRep, point_lt, point_rb  = self.fc.getImgRep(img)
            if imgRep is not False:
                imgtemp = img
                self.drawCross(imgtemp, point_lt, 25, 2, True)
                self.drawCross(imgtemp, point_rb, 25, 2, False)
                self.showMat(self.lblCamera, imgtemp) 

                self.isSearch = True
                self.txtInfo.setText("Searching start ...")                 
                for parent,  dirnames,  filenames in os.walk(self.libdir):
                    for libfilename in filenames:
                        (shortname, extension) = os.path.splitext(libfilename)
                        if '.csv' == extension:
                            self.txtInfo.append(self.libdir + '/' + libfilename)
                            distance = self.fc.compareLib(imgRep, self.libdir + '/' + libfilename)
                            if distance < tempDis:
                                tempDis = distance
                                tempPath = self.libdir + '/' + shortname + '.jpg'
                                if not os.path.exists(tempPath):
                                    tempPath = self.libdir + '/' + shortname + '.png'
                            result = "True" if distance < 0.4 else "False"
                            self.txtInfo.append("Is the same person: "+ result)
                self.isSearch = False
                if tempDis < 0.4:
                    self.txtInfo.append("Min distance : {:0.3f}".format(tempDis))
                    self.txtInfo.append("Most likely to be : " + tempPath)
                    self.showImg(self.lblResult, tempPath)     
                else:
                    self.showImg(self.lblResult, "warning.jpg")
                    self.txtInfo.append("Stranger !")
                 
                print("Searching complete .")

                if QMessageBox.question(self, "Message", "Choose whether to continue :", 
                                                    QMessageBox.Yes | QMessageBox.No, 
                                                    QMessageBox.Yes) == QMessageBox.No:
                    self.on_actionQuit_triggered()
                    
            else:
                self.txtInfo.setText("Unable to find a face !") 
                
        else:
            self.txtInfo.setText("Library is empty !") 
    
    @pyqtSlot()
    def on_actionQuit_triggered(self):
        if self.capture.isOpened():
            self.capture.release()
        self.close()
    
    @pyqtSlot()
    def on_actionAbout_triggered(self):
        QMessageBox.information(self,  "About",  "Face Verification v1.0")
    
    @pyqtSlot()
    def on_actionPhoto_triggered(self):
        self.isShow = False
        
        filename, filefilter = QFileDialog.getOpenFileNames(
            self,
            self.tr("Open File"),
            "/home/lei/project/Images/Test",
            self.tr("*.jpg *.png;;All Files(*)"),
            self.tr("*.jpg *.png"))
        if len(filename):
            self.openfilename = filename[0]
            self.txtInfo.setText(self.openfilename)
            self.showImg(self.lblCamera,  self.openfilename)
    
    @pyqtSlot()
    def on_actionLibrary_triggered(self):
        libpath = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Library Directory"),
            "/home/lei/project/Images",
            QFileDialog.Options(QFileDialog.ShowDirsOnly))
        if len(libpath):
            self.libdir = libpath
            self.refreshLib()
                    
    def refreshLib(self):
        for index in reversed(range(self.listLib.count())):
            self.listLib.takeItem(index)
        self.txtInfo.setText('')
        for parent,  dirnames,  filenames in os.walk(self.libdir):
            for filename in filenames:
                (shortname, extension) = os.path.splitext(filename)
                if '.jpg' == extension or '.png' == extension:
                    if not os.path.exists(os.path.join(parent,  shortname + ".csv")):
                        imgRep = self.fc.getRep(os.path.join(parent,  filename))
                        if imgRep is not False:
                            self.fc.saveRepToCsv(os.path.join(parent,  shortname + ".csv"), imgRep)
                    self.listLib.addItem(os.path.join(parent,  filename))       
                   
    def showImg(self, lblShow, imgpath):
        img = QPixmap(imgpath).scaledToHeight(lblShow.height())
        lblShow.setPixmap(img)
    
    def showMat(self, lblShow, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        QImg = q2n.array2qimage(img)
        QPix = QPixmap.fromImage(QImg).scaledToHeight(lblShow.height())
        lblShow.setPixmap(QPix)
    
    def showCamera(self):
        while 1:
            if not self.isShow:
                time.sleep(0.05)
            else:
                ret, img = self.capture.read()
                if camQueue.full():
                    camQueue.get()
                time.sleep(0.1)
                camQueue.put(img)
                if not camProcQueue.empty():
                    show = camProcQueue.get()
                    self.showMat(self.lblCamera, show)
                    
    def showLib(self):
        while True:
            for parent,  dirnames,  filenames in os.walk(self.libdir):
                for filename in filenames:
                    if not self.isSearch:
                        time.sleep(1)
                    else:
                        (shortname, extension) = os.path.splitext(filename)
                        if '.jpg' == extension or '.png' == extension:
                            self.showImg(self.lblResult, os.path.join(parent,  filename))
                            time.sleep(1)
                            
    def drawCross(self, img, point, length, width, leftop = True):
        y1 = y2 = point[1]
        x3 = x4 = point[0]
        if leftop:
            x1 = point[0] - length/4        
            x2 = point[0] + length
            y3 = point[1] - length/4
            y4 = point[1] + length
        else:
            x1 = point[0] - length        
            x2 = point[0] + length/4
            y3 = point[1] - length
            y4 = point[1] + length/4            
        cv2.line(img, (int(x1), y1), (int(x2), y2), (196, 208, 108), width)
        cv2.line(img, (x3, int(y3)), (x4, int(y4)), (196, 208, 108), width)

    def initCamera(self):
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened(): 
            camproc = CamProcess()
            camproc.start()
            
            self.isShow = True
            camThread = threading.Thread(target = self.showCamera)
            camThread.setDaemon(True)
            camThread.start()
        else:
            self.isShow = False
            self.txtInfo.setText('Camera Init Failed !')  
            
    def initFaceCompare(self):
        self.fc = FaceCompare()
        
class CamProcess(Process):
    def __init__(self):
        Process.__init__(self)
        self.daemon = True
        self.fc = FaceCompare()
    
    def run(self):
        while 1:
            if not camQueue.empty():
                img = camQueue.get()
                imgRep, point_lt, point_rb  = self.fc.getImgRep(img, True)
                if imgRep is not False:
                    self.drawCross(img, point_lt, 25, 2, True)
                    self.drawCross(img, point_rb, 25, 2, False)
                if camProcQueue.full():
                    camProcQueue.get()
                        
                camProcQueue.put(img)
            else:
                time.sleep(0.01)



