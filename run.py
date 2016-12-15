
import os
from Pipeline import Pipeline


def update_progress(progress,  max_steps):
    workdone = float(progress)/float(max_steps)
    #print('\r[{0}] {1}%'.format('█'*(int(100 *progress/max_steps)), progress))
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)
    

if __name__ == '__main__':
    # offline feature extraction
    #p.extractFeatures(img_db_dir='./DB/rafd2-frontal/', csv_out_filename='./DB/feat-db.csv')
  
    _GENERATE_DB = False
    _DEBUG = True
    
    frontal_sequences = ['P1E_S1_C1',  'P1E_S2_C2',  'P1E_S3_C3',  'P1E_S4_C1', 'P1L_S1_C1', 'P1L_S2_C2', 'P1L_S3_C3', 
                        'P1L_S4_C1', 'P2E_S1_C3', 'P2E_S2_C2', 'P2E_S3_C1', 'P2E_S4_C2',  'P2L_S1_C1', 'P2L_S2_C2', 'P2L_S3_C3', 'P2L_S4_C2']
    
    chokepoint_dir = "./in/" #chokepoint/
    all_sequences =  [dir for dir in os.listdir(chokepoint_dir) if os.path.isdir(os.path.join(chokepoint_dir, dir)) and dir.startswith('P')]
    
    model_path = './generator/output/FaceGen.RaFD.model.d6.adam.iter500.h5'
    feat_db_path = './DB/feat-db.csv'
    
    p = Pipeline(feat_db_path, model_path)
    print("DE-ID Pipeline started.")
    for i, seq in enumerate(frontal_sequences):
        img_dir_in = os.path.join("./in/", seq)
        img_dir_out = os.path.join("./out/", seq)
        annotation_file = os.path.join("./in/groundtruth/", (seq+'xml'))
        p.processSequence(img_dir_in, img_dir_out, _GENERATE_DB, _DEBUG)
        update_progress(i, len(frontal_sequences))
    
#    for i, seq in enumerate(all_sequences):
#        if seq not in frontal_sequences:
#            #TODO
#            pass
    
    if _DEBUG:
        cv2.destroyAllWindows()
