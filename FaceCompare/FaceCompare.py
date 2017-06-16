# -*- coding: utf-8 -*-

import time

import cv2

import os
import os.path

import numpy as np

import openface

class Args():
    def __init__(self):
        self.dlibFacePredictor = ''
        self.networkModel = ''
        self.imgDim = 96
        self.verbose = True        

class FaceCompare():
    def __init__(self):
        np.set_printoptions(precision=2)
        
        #fileDir = os.path.dirname(os.path.realpath(__file__))
        fileDir = '/home/lei/openface/demos'
        modelDir = os.path.join(fileDir, '..', 'models')
        dlibModelDir = os.path.join(modelDir, 'dlib')
        openfaceModelDir = os.path.join(modelDir, 'openface')

        self.args = Args()
        self.args.dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.args.networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
        self.args.imgDim = 96
        self.args.verbose = True

        start = time.time()
        self.align = openface.AlignDlib(self.args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.args.networkModel, self.args.imgDim)
        if self.args.verbose:
            print("Loading the dlib and OpenFace models took {} seconds.".format(
                time.time() - start))       
        
    def getRep(self,  imgPath):
        if self.args.verbose:
            print("Processing {}.".format(imgPath))
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            #raise Exception("Unable to load image: {}".format(imgPath))
            return False
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            #raise Exception("Unable to find a face: {}".format(imgPath))
            return False
        if self.args.verbose:
            print("  + Face detection took {} seconds.".format(time.time() - start))

        start = time.time()
        alignedFace = self.align.align(self.args.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            #raise Exception("Unable to align image: {}".format(imgPath))
            return False
        if self.args.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = self.net.forward(alignedFace)
        if self.args.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")

        return rep

    def getImgRep(self, img, onlyFace = False):
        if self.args.verbose:
            print("Processing Image.")
        bgrImg = img
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            #raise Exception("Unable to find a face: Image")
            return [False, (0, 0), (0, 0)]
        if self.args.verbose:
            print("  + Face detection took {} seconds.".format(time.time() - start))
        if onlyFace:
            return [True, (bb.left(), bb.top()), (bb.right(), bb.bottom())]

        start = time.time()
        alignedFace = self.align.align(self.args.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            #raise Exception("Unable to align image: Image")
            return [False, (0, 0), (0, 0)]
        if self.args.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = self.net.forward(alignedFace)
        if self.args.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")
        #csvPath = os.path.dirname(imgPath)
        #(shortname, extension) = os.path.splitext(os.path.basename(imgPath))
        #csvfile = csvPath + '/' + shortname + '.csv'
        #self.saveRepToCsv(csvfile,  rep)
        return [rep, (bb.left(), bb.top()), (bb.right(), bb.bottom())]

    def compare(self,  img1,  img2):
        img1Rep, point_lt, point_rb  = self.getImgRep(img1)
        d = img1Rep - self.getRep(img2)
        distance = np.dot(d, d)
        print("Comparing {} with {}.".format(img1, img2))
        print(
            "  + Squared l2 distance between representations: {:0.3f}".format(distance))
        return "Is the same person: {}".format(True if distance < 0.7 else False)
        
    def compareLib(self, imgRep, imgLib):
        d = imgRep - self.loadCsvToRep(imgLib)
        distance = np.dot(d, d)
        print("Comparing srcImg with {}.".format(imgLib))
        print(
            "  + Squared l2 distance between representations: {:0.3f}".format(distance))
        return distance
    
    def saveRepToCsv(self,  filename,  rep):
        np.savetxt(filename, rep, delimiter = ',')  
        
    def loadCsvToRep(self,  filename):
        rep = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0) 
        return rep
