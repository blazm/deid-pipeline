import sys, getopt
import numpy as np
import cv2

from os.path import realpath, normpath
from os.path import join

class Detector:
    
    def __init__(self):
        self.rois = []
       
        
    def detect(self,  img,  _debug=False):  
        
        self.rois = [] # reset rois if we perform multiple detections
        
        haarpath = normpath(realpath(cv2.__file__) + '../../../../../share/OpenCV/haarcascades/')
        face_cascade = cv2.CascadeClassifier(join(haarpath, 'haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(join(haarpath, 'haarcascade_eye.xml'))
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            
            self.rois.append((x, y, w, h))
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if _debug:
        #    cv2.namedWindow("Detector")
            cv2.imshow('Detected face / eyes', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        #    cv2.destroyAllWindows()
        
        return self.rois
    
    def __str__(self):
        return "{} - found faces: {}.".format(self.__class__.__name__, len(self.rois)) 
