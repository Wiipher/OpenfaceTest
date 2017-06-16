# -*- coding: utf-8 -*-

import pyCamera

class Camera():
    def __init__(self):
        self.cam = pyCamera.Camera()
        if self.cam.cameraInit():
            self.init = True
        else:
            self.init = False
            
    def free(self):
        if self.init:
            self.cam.cameraFree()
            
    def getImg(self):
        img = self.cam.cameraGetImg()
        return img

