# -*- coding: utf-8 -*-

import pyCamera
import cv2

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

cam = Camera()
print(cam.init)

for i in range(1000):
	img = cam.getImg()
	cv2.imshow("pyCamera", img)
	cv2.waitKey(5)

cam.free()
