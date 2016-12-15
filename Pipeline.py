import os
from scipy import misc
import copy
import numpy as np

import scipy.misc
import cv2 

# to view images
#from matplotlib import pyplot as plt

#--------- pipeline imports --------------
# from folder.file import class
from detector.Detector import Detector
from extractor.Extractor import Extractor
from generator.Generator import Generator
from matcher.Matcher import Matcher
from replacer.Replacer import Replacer
#--------- pipeline imports --------------

class Pipeline:
    
    def __init__(self, feat_db_path,  model_path):
        
        feat_db = np.genfromtxt(feat_db_path, skip_header=1, skip_footer=0, delimiter=',', dtype='float32', filling_values=0)
         #feat_db_file = "./DB/feat-db.csv"
        #feat_db = np.genfromtxt(feat_db_file, skip_header=1, skip_footer=0, delimiter=',', dtype='float32', filling_values=0)
        #print(len(feat_db[1,  :])) # first subject (first row)
        
        self.d = Detector()
        self.e = Extractor(include_top=True)
        self.m = Matcher(feat_db)
        self.g = Generator(model_path)
        self.r = Replacer()
    
    def resize_roi_2x(self, roi):
        '''VGG Extractor works better if the whole face image is passed to the net, so here we resize roi by factor of 2'''
        x, y, w, h = roi
        x = int(x - w/2)
        y = int(y - h/2)
        w = 2*w
        h = 2*h
        
        # TODO: check if dimensions still inside image!
        
        return (x, y, w, h)
    
        
    def extractFeatures(self,  img_db_dir='./DB/rafd2-frontal/', csv_out_filename='./DB/feat-db.csv'):
        '''Offline method to extract features from rafd2 database using VGG Extractor. Results are exported in the .csv file '''
        from os import listdir
        from os.path import isfile, join
        
        # Offline steps to produce FeatDB from Rafdb
        #e = Extractor(include_top=True)
        #d = Detector()
        
        # TODO: get list of face images in db
        # for each face extract features & save them to new text based feature db

        #img_db_dir = "./DB/rafd2-frontal/"
        file_list = [f for f in listdir(img_db_dir) if isfile(join(img_db_dir, f))]
        
        # only use frontal neutral images (for correct index to ID conversion)
        file_list = [f for f in file_list if "neutral" in f]
        
        feat_db = None
        #feat_db = np.empty((0,3),  np.float32)
        
        for file_name in file_list:
            img = misc.imread(join(img_db_dir, file_name))
            detections = self.d.detect(img,  _debug=False)
            # personID = parse.parse(rafd_name_format, file_name)[1]
            personID = int(file_name.split('_')[1])
            
            print("PersonID: {}".format(personID))
            
            if (len(detections) == 0):
                print("ERROR: Face was not detected! {}".format(file_name))
            else:
                # we can assume that only one face is in each Rafdb image
                #print(detections)
                x, y, w, h = detections[0]
                # TODO: resize to 2x the width / height 
                x, y, w, h = self.resize_roi_2x((x, y, w, h))
                
                sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
                features = self.e.extract(sub_img)
                features = np.insert(features, 0,  personID) # add person ID
                #features = np.hstack((np.array([personID]),  features));
                #print("Features_shape: {}",  (features.shape))
                
                if (feat_db is None): feat_db = features
                else: feat_db = np.vstack((feat_db,  features))
        
        print("FeatDB shape: {}", (feat_db.shape))
        
        #csv_out_filename = './DB/feat-db.csv'
        np.savetxt(csv_out_filename,  feat_db,  
                    fmt='%.9f',  delimiter=',',  
                    newline='\n',  footer='end of file',  
                    comments='# ', header='ID, features (Generated featDB from Rafdb using DeepFace VGG)')
         
    def generateNonFrontals(self, img_dir_in, img_dir_out):
        pass
         
    def loadAnnotations(self, annotation_file):
        pass
        
    def processSequence(self, img_dir_in, img_dir_out, _GENERATE_DB, _DEBUG):
        
        #img_dir_in = "./in/P1E_S1_C1/"
        #img_dir_out = "./out/P1E_S1_C1/"
        if not os.path.exists(img_dir_out):
            os.makedirs(img_dir_out)
        if not os.path.exists(os.path.join(img_dir_out,  'raw_detection')):
            os.makedirs(os.path.join(img_dir_out,  'raw_detection'))
        if not os.path.exists(os.path.join(img_dir_out,  'resize_roi_2x')):
            os.makedirs(os.path.join(img_dir_out,  'resize_roi_2x'))
        if not os.path.exists(os.path.join(img_dir_out,  'de_identified')):
            os.makedirs(os.path.join(img_dir_out,  'de_identified'))
        
        
        images = []
        for file in os.listdir(img_dir_in):
            if file.endswith(".jpg"):
                images.append(file)

        #print(images)
        
        for img_name in sorted(images):
            img = misc.imread(os.path.join(img_dir_in, img_name))
            
            #img = misc.imread('./in/people3.jpg')
            #img = misc.imread('./in/P1E_S1_C1/00001154.jpg')
            #img = misc.imread('./in/P1E_S1_C1/00001333.jpg')
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
            
            src_img = copy.copy(img) # copy orig img for Detector
            #src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
            
            if _DEBUG:
                cv2.imshow('Image input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                #cv2.waitKey(1)
            #debug
            #gauss = r.generateGaussianMask(4)
            #print("Gauss kernel:")
            
            # TODO: for loop for all images in ./in/ directory OR all frames in provided video
             # detections format - list of tuples: [(x,y,w,h), ...]
            detections = self.d.detect(src_img,  _debug=_DEBUG)
            
            #print("Detections length: {}".format(len(detections)))
            ix = 0
            for x, y, w, h in detections:
                #print(x, y, w, h)
                
                sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
                
                if _GENERATE_DB:
                    misc.toimage(sub_img, cmin=0.0, cmax=255.0).save(os.path.join(img_dir_out, "raw_detection", ("{}_".format(ix) + img_name)))
            
                
                x2, y2, w2, h2 = self.resize_roi_2x((x, y, w, h))
                
                # TODO: check bounds!
                feat_img = img[y2:y2+h2, x2:x2+w2,  :]
                
                #if _GENERATE_DB:
                #    misc.toimage(feat_img, cmin=0.0, cmax=255.0).save(os.path.join(img_dir_out, "resize_roi_2x", ("{}_".format(ix) + img_name)))
                
                #debug
                #plt.imshow(256-sub_img, interpolation='nearest')
                #plt.show()
                
                # extract features from subimage
                features = self.e.extract(feat_img)
                #print("Features shape: {}".format(features.shape))
                
                # match with best entry in feature database
                best_match = self.m.match(features)
                #print("Best match ID: {}".format(best_match))
                if best_match >= 57: # DEBUG: person_id currently limited to max 56 - Generator hardcoded identity len is 57
                    best_match = 1
                
                gen_img = self.g.generate(id=best_match)
                
                # detect face on gen img:
                gen_detection = self.d.detect(gen_img, _debug=False)
                #print("Generator detections: {}".format(gen_detection))
                
                if len(gen_detection) == 1:
                    gx, gy, gw, gh = gen_detection[0]
                    gen_img = gen_img[gy:gy+gh, gx:gx+gw, :]
                    # resize gen img to match sub_img size
                    
                    #print("GEN IMG shape before: {}".format(gen_img.shape))
                    gen_img = scipy.misc.imresize(gen_img, (w,  h,  3), interp='bicubic', mode=None)
                    
                    #print("GEN IMG shape after: {}".format(gen_img.shape))
                    
                    # TODO: replace the faces in original image
                    alt_img = self.r.replace_v2(img, (x, y, w, h),  gen_img, _debug=_DEBUG)
                
                    if _GENERATE_DB:
                        misc.toimage(alt_img[y:y+h, x:x+w, :], cmin=0.0, cmax=255.0).save(os.path.join(img_dir_out, "de_identified", ("{}_".format(ix) + img_name)))
                else:
                    alt_img = img
                    print("Warning: Number of face detections - {} - on generated image differs from 1.".format(len(gen_detection)))
                
                # DONE: swap alt_img and img for multiple detections on single image
                img = alt_img
                ix = ix+1
            
            #    cv2.namedWindow("Detector")
            #cv2.imshow('Sequence window', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
            #if _GENERATE_DB:
            #    misc.toimage(alt_img, cmin=0.0, cmax=255.0).save('./out/outfile.png')
            
        #print("De-ID pipeline: DONE.")
        
        if _DEBUG:
            cv2.destroyAllWindows()
        
    def generateProtocolTxt(self,  img_dir_out):
        # TODO: generate pairs.txt as in lfw database
        
        pass
    
    def __str__(self):
        return "{} - details: {}.".format(self.__class__.__name__, None) 
