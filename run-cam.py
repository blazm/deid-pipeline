import numpy as np
import tensorflow as tf
import time
import cv2

import os
from Pipeline import Pipeline
import cv2
from collections import defaultdict



if __name__=="__main__":

    _GENERATE_DB = False
    _DEBUG = False
    
    model_path = './generator/output/FaceGen.RaFD.model.d6.adam.iter500.h5'
    feat_db_path = './DB/feat-db.csv'
    
    p = Pipeline(feat_db_path, model_path)
    print("DE-ID Pipeline started.")
    

    cap = cv2.VideoCapture(0)
    
    i = 0
    num_detected_frames = 0
    while 1:
        ret, image = cap.read()

        if image is None or not ret:
            print("No data from camera! ")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        num_detected_frames = p.processFrame(image, i, num_detected_frames, _DEBUG=_DEBUG)

        i = i+1
        if (ret == 0):
            p.destroyWindows(_DEBUG)
            break

        #d.detect(image,True)
