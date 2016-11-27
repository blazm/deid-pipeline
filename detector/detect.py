import sys, getopt
import numpy as np
import cv2

from os.path import realpath, normpath
from os.path import join

class Detector:
    def __init__(self):
        self.rois = []
        self.haarpath = normpath(realpath(cv2.__file__) + '../../../../../share/OpenCV/haarcascades/')
        self.face_cascade = cv2.CascadeClassifier(join(haarpath, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(join(haarpath, 'haarcascade_eye.xml')
    
    def detect(self, img):            
        self.img = img
        #img = cv2.imread(inputfile)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        return self.rois
    
def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('detect.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    haarpath = normpath(realpath(cv2.__file__) + '../../../../../share/OpenCV/haarcascades/')
    face_cascade = cv2.CascadeClassifier(join(haarpath, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(join(haarpath, 'haarcascade_eye.xml'))

    img = cv2.imread(inputfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # TODO: save rectangles to file or pass them on to the extractor

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
   main(sys.argv[1:])
