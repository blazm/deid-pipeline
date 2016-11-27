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

def offline(): 
    from os import listdir
    from os.path import isfile, join
    
    # Offline steps to produce FeatDB from Rafdb
    e = Extractor(include_top=True)
    d = Detector()
    
    # TODO: get list of face images in db
    # for each face extract features & save them to new text based feature db

    img_db_dir = "./generator/rafd2-frontal/"
    file_list = [f for f in listdir(img_db_dir) if isfile(join(img_db_dir, f))]
    
    # only use frontal neutral images (for correct index to ID conversion)
    file_list = [f for f in file_list if "neutral" in f]
    
    feat_db = None
    #feat_db = np.empty((0,3),  np.float32)
    
    for file_name in file_list:
        img = misc.imread(join(img_db_dir, file_name))
        detections = d.detect(img,  _debug=False)
        # personID = parse.parse(rafd_name_format, file_name)[1]
        personID = int(file_name.split('_')[1])
        
        print("PersonID: {}".format(personID))
        
        if (len(detections) == 0):
            print("ERROR: Face was not detected! {}".format(file_name))
        else:
            # we can assume that only one face is in each Rafdb image
            #print(detections)
            x, y, w, h = detections[0]
            sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
            features = e.extract(sub_img)
            features = np.insert(features, 0,  personID) # add person ID
            #features = np.hstack((np.array([personID]),  features));
            #print("Features_shape: {}",  (features.shape))
            
            if (feat_db is None): feat_db = features
            else: feat_db = np.vstack((feat_db,  features))
    
    print("FeatDB shape: {}", (feat_db.shape))
    
    np.savetxt('./DB/feat-db.csv',  feat_db,  
                fmt='%.9f',  delimiter=',',  
                newline='\n',  footer='end of file',  
                comments='# ', header='ID, features (Generated featDB from Rafdb using DeepFace VGG)')
    
_DEBUG = True

def main():
    # Online (main) pipeline

# Legacy args handling
#    inputfile = ''
#    outputfile = ''
#    try:
#        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
#    except getopt.GetoptError:
#        print('detect.py -i <inputfile> -o <outputfile>')
#        sys.exit(2)
#        
#    for opt, arg in opts:
#        if opt == '-h':
#            print('test.py -i <inputfile> -o <outputfile>')
#            sys.exit()
#        elif opt in ("-i", "--ifile"):
#            inputfile = arg
#        elif opt in ("-o", "--ofile"):
#            outputfile = arg

    feat_db_file = "./DB/feat-db.csv"
    feat_db = np.genfromtxt(feat_db_file, skip_header=1, skip_footer=0, delimiter=',', dtype='float32', filling_values=0)
    print(len(feat_db[1,  :])) # first subject (first row)
    
    img = misc.imread('./in/people3.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
    
    src_img = copy.copy(img) # copy orig img for Detector
    #src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR) # convert color space for proper image display
    
    d = Detector()
    e = Extractor(include_top=True)
    m = Matcher(feat_db)
    # TODO: rest of the pipeline
    g = Generator(model_path='./generator/output/FaceGen.RaFD.model.d6.adam.iter500.h5')
    r = Replacer()
    
    #debug
    #gauss = r.generateGaussianMask(4)
    #print("Gauss kernel:")
    #print(gauss)
    
    # TODO: for loop for all images in ./in/ directory OR all frames in provided video
     # detections format - list of tuples: [(x,y,w,h), ...]
    detections = d.detect(src_img,  _debug=_DEBUG)
    
    print("Detections length: {}".format(len(detections)))
    for x, y, w, h in detections:
        print(x, y, w, h)
        
        sub_img = img[y:y+h, x:x+w, :] # take subarray from source image
        
        #debug
        #plt.imshow(256-sub_img, interpolation='nearest')
        #plt.show()
        
        # extract features from subimage
        features = e.extract(sub_img)
        print("Features shape: {}".format(features.shape))
        
        # match with best entry in feature database
        best_match = m.match(features)
        print("Best match ID: {}".format(best_match))
        if best_match >= 57: # DEBUG: person_id currently limited to max 56 - Generator hardcoded identity len is 57
            best_match = 1
        
        gen_img = g.generate(id=best_match)
        
        # detect face on gen img:
        gen_detection = d.detect(gen_img, _debug=_DEBUG)
        print("Generator detections: {}".format(gen_detection))
        
        if len(gen_detection) == 1:
            gx, gy, gw, gh = gen_detection[0]
            gen_img = gen_img[gy:gy+gh, gx:gx+gw, :]
            # resize gen img to match sub_img size
            
            #print("GEN IMG shape before: {}".format(gen_img.shape))
            gen_img = scipy.misc.imresize(gen_img, (w,  h,  3), interp='bilinear', mode=None)
            
            #print("GEN IMG shape after: {}".format(gen_img.shape))
            
            # TODO: replace the faces in original image
            alt_img = r.replace_v2(img, (x, y, w, h),  gen_img, _debug=_DEBUG)
            
        else:
            alt_img = img
            print("Warning: Number of face detections - {} - on generated image differs from 1.".format(len(gen_detection)))
        
        # DONE: swap alt_img and img for multiple detections on single image
        img = alt_img
    
    misc.toimage(alt_img, cmin=0.0, cmax=255.0).save('./out/outfile.png')
        
    print("De-ID pipeline: DONE.")
    

if __name__ == '__main__':
 #   offline() # To generate FeatDB from Rafdb
    main()
    
    if _DEBUG:
        cv2.destroyAllWindows()
