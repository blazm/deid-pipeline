import os
from Pipeline import Pipeline
import cv2
from collections import defaultdict

def update_progress(progress,  max_steps):
    workdone = float(progress)/float(max_steps)
    #print('\r[{0}] {1}%'.format('█'*(int(100 *progress/max_steps)), progress))
    print("\rProgress: [{0:50s}] {1:.1f}% \n".format('#' * int(workdone * 50), workdone*100), end="", flush=True)
    
def samplePairs(img_dir_out, num_pairs):
    from random import choice, sample
    from os import listdir
    from os.path import isfile, join
    filenames = [f for f in listdir(img_dir_out) if isfile(join(img_dir_out, f))]
    
    # repeat 300x10:
    #   pick random ID
    #   pick selected ID subset
    #   pick two images from subset, that are adjacent enough
    
    img_dict = defaultdict(lambda: defaultdict(lambda: [])) # default values are None and []
    print(filenames)
    for filename in filenames:
        id, seq, frame = filename.split('-') 
        frame = frame.split('.')[0]
        img_dict[id][seq].append(frame)
       
    for i in range(0, num_pairs):
        sel_id = choice(list(img_dict.keys()))
      
        sel_seqs = sample(list(img_dict[sel_id].keys()), 2)
        
        sel_frame1 = choice(list(img_dict[sel_id][sel_seqs[0]]))
        sel_frame2 = choice(list(img_dict[sel_id][sel_seqs[1]]))
        
        print(sel_id, sel_seqs, sel_frame1, sel_frame2)
        
    print(len(img_dict.keys()))
        
        # TODO: sample pairs
    pass

def sampleNonPairs(img_dir_out):
    pass    

if __name__ == '__main__':
    # offline feature extraction
    #p.extractFeatures(img_db_dir='./DB/rafd2-frontal/', csv_out_filename='./DB/feat-db.csv')
    
    _GENERATE_DB = False
    _DEBUG = True
    
    frontal_sequences = ['P1E_S1_C1',  'P1E_S2_C2', 'P1E_S3_C3', 'P1E_S4_C1', 'P1L_S1_C1', 
                        'P1L_S2_C2', 'P1L_S3_C3', 'P1L_S4_C1', 'P2E_S1_C3.1', 'P2E_S1_C3.2',  
                        'P2E_S2_C2.1', 'P2E_S2_C2.2', 'P2E_S3_C1.1', 'P2E_S3_C1.2', 
                        'P2E_S4_C2.1', 'P2E_S4_C2.2', 'P2L_S1_C1.1', 'P2L_S1_C1.2', 
                        'P2L_S2_C2.1', 'P2L_S2_C2.2', 'P2L_S3_C3.1', 'P2L_S3_C3.2', 
                        'P2L_S4_C2.1', 'P2L_S4_C2.2']
    
    chokepoint_dir = "./in/" #chokepoint/
    groundtruth_dir = "./in/groundtruth/"
    all_sequences = [dir for dir in os.listdir(chokepoint_dir) if os.path.isdir(os.path.join(chokepoint_dir, dir)) and dir.startswith('P') and os.path.exists(os.path.join("./in/groundtruth/", (dir+'.xml')))]
   # print("{}".format(all_sequences))
    model_path = './generator/output/FaceGen.RaFD.model.d6.adam.iter500.h5'
    feat_db_path = './DB/feat-db.csv'
    
    p = Pipeline(feat_db_path, model_path)
    print("DE-ID Pipeline started.")
    
    # DEBUG & TESTING
    seq = frontal_sequences[5]
    img_dir_in = os.path.join("./in/", seq)
    img_dir_out = os.path.join("./out/", seq)
    groundtruth_path = os.path.join("./in/groundtruth/", (seq+'.xml'))
    #print(groundtruth_path)
    #p.processSequence(img_dir_in, img_dir_out, groundtruth_path, _GENERATE_DB, _DEBUG)
    #p.generateGroundTruth(img_dir_in, img_dir_out, groundtruth_path, _GENERATE_DB)
    # TODO:
    #samplePairs(os.path.join(img_dir_out, 'raw_groundtruth'), 300)
       
    # FINAL RUN, when everything is ready
    if False:
        for i, seq in enumerate(frontal_sequences):
            img_dir_in = os.path.join("./in/", seq)
            #img_dir_out = os.path.join("./out/", seq)
            img_dir_out = os.path.join("./out/", "deid-eval-db")
            groundtruth_path = os.path.join("./in/groundtruth/", (seq+'.xml'))
            p.processSequence(img_dir_in, img_dir_out, groundtruth_path, _GENERATE_DB, _DEBUG)
            update_progress(i, len(frontal_sequences))
    
        # extract only non frontals from the groundtruth
        for i, seq in enumerate(all_sequences):
            if seq not in frontal_sequences:
                img_dir_in = os.path.join("./in/", seq)
                #img_dir_out = os.path.join("./out/", seq)
                img_dir_out = os.path.join("./out/", "deid-eval-db")
                groundtruth_path = os.path.join("./in/groundtruth/", (seq+'.xml'))
                p.generateGroundTruth(img_dir_in, img_dir_out, groundtruth_path, _GENERATE_DB)
            update_progress(i, len(all_sequences))
            
    samplePairs(os.path.join("./out/", "deid-eval-db",  "de_identified"), 10)
        
    if _DEBUG:
        cv2.destroyAllWindows()
