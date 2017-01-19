import os
from Pipeline import Pipeline
import cv2
from collections import defaultdict

def update_progress(progress,  max_steps):
    workdone = float(progress)/float(max_steps)
    #print('\r[{0}] {1}%'.format('█'*(int(100 *progress/max_steps)), progress))
    print("\rProgress: [{0:50s}] {1:.1f}% \n".format('#' * int(workdone * 50), workdone*100), end="", flush=True)
    
def testProtocolFiles(protocol_file_path):
    all_exist = True    
    path = os.path.split(protocol_file_path)[0]

    with open(protocol_file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')
            items = line.split('\t')
            if len(items) <= 2: continue # ignore first line
            
            if not os.path.isfile(os.path.join(path, items[1])): 
                print("2nd : ", os.path.join(path, items[1]))
                all_exist = False
            if not os.path.isfile(os.path.join(path, items[-1])):
                print("last: ", os.path.join(path, items[-1]))
                all_exist = False
    return all_exist
    
def samplePairs(img_dir_out, img_gt_dir_out, num_pairs, num_folds):
    #from random import choice, sample
    import random as r
    from os import listdir
    from os.path import isfile, join
    filenames = [f for f in listdir(img_dir_out) if isfile(join(img_dir_out, f))]
    gt_filenames = [f for f in listdir(img_gt_dir_out) if isfile(join(img_gt_dir_out, f))] # groundtruth images are from different sequences
    
    random = r
    random.seed(42)
    # repeat 300x10:
    #   pick random ID
    #   pick selected ID subset
    #   pick two images from subset, that are adjacent enough
    
    delimiter = '-'
    
    img_dict = defaultdict(lambda: defaultdict(lambda: [])) # default values are None and []
    img_gt_dict = defaultdict(lambda: defaultdict(lambda: [])) # default values are None and []
    
    #print(filenames)
    for filename in filenames:
        id, seq, frame = filename.split(delimiter) 
        frame = frame.split('.')[0]
        img_dict[id][seq].append(frame)
    
    for filename in gt_filenames:
        id, seq, frame = filename.split(delimiter) 
        frame = frame.split('.')[0]
        img_gt_dict[id][seq].append(frame)
    
    # TODO: select pairs from different folders!
    # orig - orig, orig- de_id, orig-profile, de_id-profile
    
    pairs = []
    gt_items = []
    
     # TODO: another for loop for 10 folding
    for f in range(0, num_folds):
    
        # generate matches
        for i in range(0, num_pairs):
            # intersection of ids in all sequences - make sure that ids really match
            available_ids = list(set(img_dict.keys()) & set(img_gt_dict.keys()))
            sel_id = random.choice(available_ids)
           
            sel_seqs = random.sample(list(img_dict[sel_id].keys()), 2)
            sel_seq1, sel_seq2 = sel_seqs
            
            sel_frame1 = random.choice(list(img_dict[sel_id][sel_seq1]))
            sel_frame2 = random.choice(list(img_dict[sel_id][sel_seq2]))
            
            #print(sel_id, sel_seq1, sel_seq2, sel_frame1, sel_frame2)
            img_name1 = delimiter.join((sel_id, sel_seq1, sel_frame1)) + '.jpg'
            img_name2 = delimiter.join((sel_id, sel_seq2, sel_frame2)) + '.jpg'
            pairs.append((sel_id, img_name1, img_name2))
            
            sel_gt_seq = random.choice(list(img_gt_dict[sel_id].keys()))
            sel_gt_frame = random.choice(list(img_gt_dict[sel_id][sel_gt_seq]))
            img_gt_name = delimiter.join((sel_id, sel_gt_seq, sel_gt_frame)) + '.jpg'
            gt_items.append((sel_id, img_gt_name))
            
        # generate non-matches
        for i in range(0, num_pairs):
            sel_ids = random.sample(list(img_dict.keys()), 2)
            sel_id1, sel_id2 = sel_ids
            
            sel_seq1 = random.choice(list(img_dict[sel_id1].keys()))
            sel_seq2 = random.choice(list(img_dict[sel_id2].keys()))
            
            sel_frame1 = random.choice(list(img_dict[sel_id1][sel_seq1]))
            sel_frame2 = random.choice(list(img_dict[sel_id2][sel_seq2]))
            
            img_name1 = delimiter.join((sel_id1, sel_seq1, sel_frame1)) + '.jpg'
            img_name2 = delimiter.join((sel_id2, sel_seq2, sel_frame2)) + '.jpg'
            pairs.append((sel_id1, img_name1, sel_id2, img_name2))
            
            gt_ids = list(img_gt_dict.keys())
            gt_ids.remove(sel_id1)
            gt_ids.remove(sel_id2)
            
            sel_gt_id = random.choice(gt_ids)
            sel_gt_seq = random.choice(list(img_gt_dict[sel_gt_id].keys()))
            sel_gt_frame = random.choice(list(img_gt_dict[sel_gt_id][sel_gt_seq]))
            img_gt_name = delimiter.join((sel_gt_id, sel_gt_seq, sel_gt_frame)) + '.jpg'
            gt_items.append((sel_gt_id, img_gt_name))
            
    combos = [('orig-orig', 'raw_detection', 'raw_detection'), 
                ('orig-deid', 'raw_detection', 'de_identified')] 
    combos_gt = [('orig-profile', 'raw_detection', 'raw_groundtruth'), 
                 ('deid-profile', 'de_identified', 'raw_groundtruth')]
        
    pairs_path = os.path.relpath(os.path.join(img_dir_out, '..'))
    
    for prefix, dir1, dir2 in combos:
        
        pairs_filename = os.path.join(pairs_path, "_".join((prefix,'pairs.txt')))
        pairs_file = open(pairs_filename, 'w')
        pairs_file.write("%s\t%s\n" % (num_folds, num_pairs))
        for pair_tuple in pairs:
        
            l = list(pair_tuple)
            l[1] = os.path.join('.', dir1, l[1])
            l[-1] = os.path.join('.', dir2, l[-1])
            pair = tuple(l)
    
            #for item in pair_tuple:
            
            pair_line = "\t".join(pair)
            pairs_file.write("%s\n" % pair_line)
        pairs_file.close()
        
        if not testProtocolFiles(pairs_filename):
            print("ERROR: Not all items exist in pairs file: ",  pairs_filename)
        else:
            print("SUCCESS: All items exist in pairs file: ",  pairs_filename)
            
    for prefix, dir1, dir2 in combos_gt:
        pairs_filename = os.path.join(pairs_path, "_".join((prefix,'pairs.txt')))
        pairs_file = open(pairs_filename, 'w')
        pairs_file.write("%s\t%s\n" % (num_folds, num_pairs))
        for pair_tuple, gt_tuple in zip(pairs, gt_items):
        
            gt_pair = pair_tuple[:2] + (gt_tuple[-1],)
            if (pair_tuple[0] != gt_tuple[0]):
                gt_pair = pair_tuple[:2] + gt_tuple

            l = list(gt_pair)
            l[1] = os.path.join('.', dir1, l[1])
            l[-1] = os.path.join('.', dir2, l[-1])
            pair = tuple(l)

            pair_line = "\t".join(pair)
            pairs_file.write("%s\n" % pair_line)
        pairs_file.close()
        
        if not testProtocolFiles(pairs_filename):
            print("ERROR: Not all items exist in pairs file: ",  pairs_filename)
        else:
            print("SUCCESS: All items exist in pairs file: ",  pairs_filename)

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
    
    if _DEBUG:
        seq = frontal_sequences[5]
        img_dir_in = os.path.join("./in/", seq)
        img_dir_out = os.path.join("./out/", seq)
        groundtruth_path = os.path.join("./in/groundtruth/", (seq+'.xml'))
        #print(groundtruth_path)
        p.processSequence(img_dir_in, img_dir_out, groundtruth_path, _GENERATE_DB, _DEBUG)
        p.generateGroundTruth(img_dir_in, img_dir_out, groundtruth_path, _GENERATE_DB)
        # TODO:
        #samplePairs(os.path.join(img_dir_out, 'raw_groundtruth'), 300)
       
    # FINAL RUN, when everything is ready
    if not _DEBUG:
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
            
        samplePairs(os.path.join("./out/", "deid-eval-db",  "de_identified"), os.path.join("./out/", "deid-eval-db",  "raw_groundtruth"), num_pairs=300, num_folds=10)
        
    if _DEBUG:
        cv2.destroyAllWindows()
