import os
from collections import defaultdict

import copy
import numpy as np

from scipy.misc import imresize
from scipy.ndimage import imread

import cv2 

# to view images
#from matplotlib import pyplot as plt

#--------- pipeline imports --------------
# from folder.file import class

#from detector2.Detector import Detector # using SSD
from detector.Detector import Detector # using ViolaJones
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
        
        return (x, y, w, h)
        
    def crop_from(self, img, roi):
        x, y, w, h = roi
        sx, sy, sw, sh = roi         
        ih, iw, ch = img.shape
        # TODO: check borders
        #sub_img = np.zeros((h, w, ch), dtype=img.dtype)
        
        dx = 0
        if (x < 0):
            dx = abs(x)
            x = 0
            w = w - dx
        elif (x+w > iw):
            w = iw - x
            
        dy = 0
        if (y < 0):
            dy = abs(y)
            y = 0
            h = h - dy
        elif (y+h > ih):
            h = ih - y
        
        return img[y:y+h, x:x+w, :] # take subarray from source image
        
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
            img = imread(join(img_db_dir, file_name))
            ih, iw, ch = img.shape
            
            detections = self.d.detect(img,  _debug=False)
            # person_id = parse.parse(rafd_name_format, file_name)[1]
            person_id = int(file_name.split('_')[1])
            
            print("PersonID: {}".format(person_id))
            
            if (len(detections) == 0):
                print("ERROR: Face was not detected! {}".format(file_name))
            else:
                # we can assume that only one face is in each Rafdb image
                #print(detections)
                x, y, w, h = detections[0]
                # TODO: resize to 2x the width / height 
                x, y, w, h = self.resize_roi_2x((x, y, w, h))
                
                #sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
                sub_img = self.crop_from(img, (x, y, w, h))
                
                features = self.e.extract(sub_img)
                features = np.insert(features, 0,  person_id) # add person ID
                #features = np.hstack((np.array([person_id]),  features));
                #print("Features_shape: {}",  (features.shape))
                
                if (feat_db is None): feat_db = features
                else: feat_db = np.vstack((feat_db,  features))
        
        print("FeatDB shape: {}", (feat_db.shape))
        
        #csv_out_filename = './DB/feat-db.csv'
        np.savetxt(csv_out_filename,  feat_db,  
                    fmt='%.9f',  delimiter=',',  
                    newline='\n',  footer='end of file',  
                    comments='# ', header='ID, features (Generated featDB from Rafdb using DeepFace VGG)')
         
    def generateGroundTruth(self, img_dir_in, img_dir_out, groundtruth_xml_path, _GENERATE_DB = True):
        if not os.path.exists(img_dir_out):
            os.makedirs(img_dir_out)
        if not os.path.exists(os.path.join(img_dir_out,  'raw_groundtruth')):
            os.makedirs(os.path.join(img_dir_out,  'raw_groundtruth'))
        
        sequence_name = os.path.split(img_dir_in)[1]
        #print(sequence_name)
        groundtruth_dict = self.loadGroundTruthXML(groundtruth_xml_path)
        #print(groundtruth_dict.keys())
        images = []
        for file in os.listdir(img_dir_in):
            if file.endswith(".jpg"):
                images.append(file)
        #print("{}".format(len(images)))
        for img_name in sorted(images):
            frame_number = img_name.split('.')[0]
            if int(frame_number) % 5 != 0: continue
            
            #print(frame_number)
            annotated_people = groundtruth_dict[frame_number]
            if not annotated_people: continue
            #print("Annotated: {}".format(annotated_people))
            
            img = imread(os.path.join(img_dir_in, img_name))
        
            for person in annotated_people:
                person_id, face_roi = person
                sub_img = self.crop_from(img, face_roi)
                
                if _GENERATE_DB:
                    misc.toimage(sub_img, cmin=0.0, cmax=255.0).save(
                    os.path.join(img_dir_out, "raw_groundtruth", ("{}-{}-".format(person_id, sequence_name) + img_name)))
         
    def loadGroundTruthXML(self, groundtruth_xml_path):
        import xml.etree.ElementTree as ET
        from math import sqrt, ceil

        def bb_from_xy(x1, y1, x2, y2):
            '''Input: two eye coords, Output: face ROI'''
            x = int((x1 + x2)/2.)
            y = int((y1 + y2)/2.)
            d = int(sqrt((x1-x2)**2 + (y1-y2)**2))
            return (x-ceil(1.5*d),  y-ceil(1.5*d), 3*d, ceil(3.5*d))
        
        f = open(groundtruth_xml_path, 'r')
        data = f.read()
        f.close()
        
        root = ET.fromstring(data) 
        frame_dict = defaultdict(lambda: []) # default values are empty lists
        
        for frame in root.findall('./frame'):
            frame_num = frame.attrib.get('number')
            
            persons = []
            for person in frame.findall('./person'):            
                if person is None: continue
                
                person_id = person.attrib.get('id')
                
                try:
                    leftEye = person.find('leftEye')
                    #if leftEye:
                    xl, yl = int(leftEye.attrib.get('x')), int(leftEye.attrib.get('y'))
                    rightEye = person.find('rightEye')
                    #if rightEye:
                    xr, yr = int(rightEye.attrib.get('x')), int(rightEye.attrib.get('y'))

                    #if rightEye and leftEye:
                    bb = bb_from_xy(xl, yl, xr, yr)
                    persons.append((person_id,  bb))
                except:
                    import sys
                    print("ERROR: Detector - detect - unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
            
            if persons:
                frame_dict[frame_num] = persons
            
        return frame_dict
        
    def overlappingRatio(self,  roiA,  roiB):
        '''Returns overlapping ration of two ROIs (x,y,w,h)'''
        xA, yA, wA, hA = roiA
        xB, yB, wB, hB = roiB
        SI = max(0, min(xA+wA, xB+wB) - max(xA, xB)) * max(0, min(yA+hA, yB+hB) - max(yA, yB))
        SA = wA*hA
        SB = wB*hB
        SU = SA + SB - SI
        return SI / SU
        
    def processSequence(self, img_dir_in, img_dir_out, groundtruth_xml_path, _GENERATE_DB, _DEBUG, _LOAD_GT):
        sequence_name = os.path.split(img_dir_in)[1]
        
        if _LOAD_GT:
            groundtruth_dict = self.loadGroundTruthXML(groundtruth_xml_path)
        
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

        num_detected_frames = 0

        for img_name in sorted(images):
            frame_number = img_name.split('.')[0]
            #if int(frame_number) % 5 != 0: continue # take just one image per second
            
            if _LOAD_GT:            
                annotated_people = groundtruth_dict[frame_number]
                if not annotated_people: continue
                
            img = imread(os.path.join(img_dir_in, img_name))
            img = imresize(img, (img.shape[0]//2,  img.shape[1]//2,  3), interp='bicubic', mode=None)
                    
            #for person in annotated_people:
            #    person_id, face_roi = person
            
            #img = misc.imread('./in/people3.jpg')
            #img = misc.imread('./in/P1E_S1_C1/00001154.jpg')
            #img = misc.imread('./in/P1E_S1_C1/00001333.jpg')
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
            
           # src_img = copy.copy(img) # copy orig img for Detector
            #src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
            
            #if _DEBUG:
             #   cv2.imshow('Image input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                #cv2.waitKey(1)
            
            # TODO: for loop for all images in ./in/ directory OR all frames in provided video
            # detections format - list of tuples: [(x,y,w,h), ...]
            detections = self.d.detect(img,  _debug=_DEBUG)
            
            if len(detections) > 0:
                num_detected_frames = num_detected_frames+1;
            
            #print("Detections length: {}".format(len(detections)))
            for ix, roi in enumerate(detections):
                x, y, w, h = roi
                
                
                if _LOAD_GT:
                    # find most overlapped groundtruth ID (if there are multiple people in sequence)
                    max_ratio = 0
                    selected_person_id = None
                    for person in annotated_people:
                        person_id, face_roi = person
                        r = self.overlappingRatio(roi,  face_roi)
                        if r > max_ratio:
                            max_ratio = r
                            selected_person_id = person_id
                    
                    if not selected_person_id:
                        print("Warning: personID not found for frame {} in detection {}!".format(frame_number, ix))
                        continue
                else:
                    selected_person_id = 0;
                #print(x, y, w, h)
                
                #sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
                sub_img = self.crop_from(img, (x, y, w, h))
                
                if _GENERATE_DB:
                    misc.toimage(sub_img, cmin=0.0, cmax=255.0).save(
                    os.path.join(img_dir_out, "raw_detection", ("{}-{}-".format(selected_person_id, sequence_name) + img_name)))
                
                x2, y2, w2, h2 = self.resize_roi_2x((x, y, w, h))
                
                # TODO: check bounds!
                #feat_img = img[y2:y2+h2, x2:x2+w2,  :]
                feat_img = self.crop_from(img, (x2, y2, w2, h2))
                
                if _GENERATE_DB:
                    misc.toimage(feat_img, cmin=0.0, cmax=255.0).save(
                    os.path.join(img_dir_out, "resize_roi_2x", ("{}-{}-".format(selected_person_id, sequence_name) + img_name)))
                
                # extract features from subimage
                features = self.e.extract(feat_img)
                #print("Features shape: {}".format(features.shape))
                
                # match with best entry in feature database
                max_id_limit = 57 # DEBUG: person_id currently limited to max 56 - Generator hardcoded identity len is 57
                best_match = self.m.match(features, max_id_limit, selected_person_id)
                #print("Best match ID: {}".format(best_match))
                if best_match >= 57:
                    print("Warning: BestMatch exceeds maximum - {}".format(best_match))
                    best_match = 1
                
                
                gen_img = self.g.generate(id=best_match)
                
                # detect face on gen img:
                gen_detection = self.d.detect(gen_img, _debug=False)
                #print("Generator detections: {}".format(gen_detection))
                
                if len(gen_detection) == 1:
                    gx, gy, gw, gh = gen_detection[0] # TODO: resize roi to cover full face?
                    
                    #gen_img = gen_img[gy:gy+gh, gx:gx+gw, :]
                    gen_img = self.crop_from(gen_img, (gx, gy, gw, gh))
              
                    # resize gen img to match sub_img size
                    
                    #print("GEN IMG shape before: {}".format(gen_img.shape))
                    gen_img = imresize(gen_img, (w,  h,  3), interp='bicubic', mode=None)
                    
                    #print("GEN IMG shape after: {}".format(gen_img.shape))
                    
                    # TODO: replace the faces in original image
                    try:
                        alt_img = self.r.replace_v3(img, (x, y, w, h), gen_img, _debug=_DEBUG)
                    except:
                        alt_img = img
                        print("SRC ROI: {}".format((x, y, w, h)))
                        print("GENERATED: {}".format(gen_img.shape))
                        import sys
                        print("ERROR: Replacer - replace - unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
            
                
                    if _GENERATE_DB:
                        #db_img = alt_img[y:y+h, x:x+w, :]
                        db_img = self.crop_from(alt_img, (x, y, w, h))
                        misc.toimage(db_img, cmin=0.0, cmax=255.0).save(
                        os.path.join(img_dir_out, "de_identified",  ("{}-{}-".format(selected_person_id, sequence_name) + img_name)))
                else:
                    alt_img = img
                    print("Warning: Number of face detections - {} - on generated image differs from 1.".format(len(gen_detection)))
                
                # DONE: swap alt_img and img for multiple detections on single image
                img = alt_img

            if _DEBUG:
                cv2.waitKey(1)
            # TODO: save whole frames fro article images
        
            #    cv2.namedWindow("Detector")
            #cv2.imshow('Sequence window', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
            #if _GENERATE_DB:
            #misc.toimage(src_img, cmin=0.0, cmax=255.0).save('./out/' + "orig-{}-".format(sequence_name) + img_name)
            #misc.toimage(img, cmin=0.0, cmax=255.0).save('./out/' + "deid-{}-".format(sequence_name) + img_name)
            #misc.toimage(alt_img, cmin=0.0, cmax=255.0).save('./out/outfile.png')
            
        #print("De-ID pipeline: DONE.")
        
        if _DEBUG:
            cv2.destroyAllWindows()
            
        return num_detected_frames
        
    def generateProtocolTxt(self,  img_dir_out):
        # TODO: generate pairs.txt as in lfw database
        
        pass


    def processFrame(self, img, frame_number, num_detected_frames, _DEBUG=True):

        _DEBUG = True
        _DEMO = True
        #frame_number = img_name.split('.')[0]
        #if int(frame_number) % 5 != 0: continue # take just one image per second
        _GENERATE_DB = False
        #if _LOAD_GT:            
        #    annotated_people = groundtruth_dict[frame_number]
        #    if not annotated_people: continue
            
        #img = imread(os.path.join(img_dir_in, img_name))
        #
        img = imresize(img, (img.shape[0]//2,  img.shape[1]//2,  3), interp='bicubic', mode=None)
                
        #for person in annotated_people:
        #    person_id, face_roi = person
        
        #img = misc.imread('./in/people3.jpg')
        #img = misc.imread('./in/P1E_S1_C1/00001154.jpg')
        #img = misc.imread('./in/P1E_S1_C1/00001333.jpg')
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
        
        

        src_img = copy.copy(img) # copy orig img for Detector
        #src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
        alt_img = img
        #if _DEBUG:
         #   cv2.imshow('Image input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
        
        # TODO: for loop for all images in ./in/ directory OR all frames in provided video
        # detections format - list of tuples: [(x,y,w,h), ...]
        detections = self.d.detect(img,  _debug=False)
        
        if len(detections) > 0:
            num_detected_frames = num_detected_frames+1;
        
        #print("Detections length: {}".format(len(detections)))
        for ix, roi in enumerate(detections):
            x, y, w, h = roi
            
            
            selected_person_id = 0;
            #print(x, y, w, h)
            
            #sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
            sub_img = self.crop_from(img, (x, y, w, h))
            
            if _GENERATE_DB:
                misc.toimage(sub_img, cmin=0.0, cmax=255.0).save(
                os.path.join(img_dir_out, "raw_detection", ("{}-{}-".format(selected_person_id, sequence_name) + img_name)))
            
            x2, y2, w2, h2 = self.resize_roi_2x((x, y, w, h))
            
            # TODO: check bounds!
            #feat_img = img[y2:y2+h2, x2:x2+w2,  :]
            feat_img = self.crop_from(img, (x2, y2, w2, h2))
            
            if _GENERATE_DB:
                misc.toimage(feat_img, cmin=0.0, cmax=255.0).save(
                os.path.join(img_dir_out, "resize_roi_2x", ("{}-{}-".format(selected_person_id, sequence_name) + img_name)))
            
            # extract features from subimage
            features = self.e.extract(feat_img)
            #print("Features shape: {}".format(features.shape))
            
            # match with best entry in feature database
            max_id_limit = 57 # DEBUG: person_id currently limited to max 56 - Generator hardcoded identity len is 57
            best_match = self.m.match(features, max_id_limit, selected_person_id)
            #print("Best match ID: {}".format(best_match))
            if best_match >= 57:
                print("Warning: BestMatch exceeds maximum - {}".format(best_match))
                best_match = 1
            
            
            gen_img = self.g.generate(id=best_match)
            
            # detect face on gen img:
            gen_detection = self.d.detect(gen_img, _debug=False)
            #print("Generator detections: {}".format(gen_detection))
            
            if len(gen_detection) == 1:
                gx, gy, gw, gh = gen_detection[0] # TODO: resize roi to cover full face?
                
                #gen_img = gen_img[gy:gy+gh, gx:gx+gw, :]
                gen_img = self.crop_from(gen_img, (gx, gy, gw, gh))
          
                # resize gen img to match sub_img size
                
                #print("GEN IMG shape before: {}".format(gen_img.shape))
                gen_img = imresize(gen_img, (w,  h,  3), interp='bicubic', mode=None)
                
                #print("GEN IMG shape after: {}".format(gen_img.shape))
                
                # TODO: replace the faces in original image
                try:
                    #alt_img = self.r.replace_v3(img, (x, y, w, h), gen_img, _debug=_DEBUG)
                    alt_img = self.r.replace_v3(img, (x, y, w, h), gen_img, _debug=False)
                except:
                    alt_img = img
                    print("SRC ROI: {}".format((x, y, w, h)))
                    print("GENERATED: {}".format(gen_img.shape))
                    import sys
                    print("ERROR: Replacer - replace - unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
        
            
                if _GENERATE_DB:
                    #db_img = alt_img[y:y+h, x:x+w, :]
                    db_img = self.crop_from(alt_img, (x, y, w, h))
                    misc.toimage(db_img, cmin=0.0, cmax=255.0).save(
                    os.path.join(img_dir_out, "de_identified",  ("{}-{}-".format(selected_person_id, sequence_name) + img_name)))
            else:
                alt_img = img
                print("Warning: Number of face detections - {} - on generated image differs from 1.".format(len(gen_detection)))
            
            # DONE: swap alt_img and img for multiple detections on single image
            img = alt_img



        if _DEMO:
        
            final_img = np.hstack((src_img, alt_img))
        
            cv2.imshow("Source Image / Output Image", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)

        return num_detected_frames


    def destroyWindows(_DEBUG=True):
        if _DEBUG:
            cv2.destroyAllWindows()
    
    def __str__(self):
        return "{} - details: {}.".format(self.__class__.__name__, None) 
