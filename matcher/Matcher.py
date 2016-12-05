
#from scipy.spatial.distance import cosine
import numpy as np

class Matcher:
    
    def __init__(self,  feat_db):
        # TODO: load DB?
        self.feat_db = feat_db
        self.best_match = 0
        self.best_id = None
        
    def cosine(self, a, b):
        return np.dot(a,b.T)/np.linalg.norm(a)/np.linalg.norm(b)
        
    def match_n(self, feats):
        pass

    def match(self,  feats):  
        self.best_match = 0
        self.best_id = None
        for i in range(self.feat_db.shape[0]): # number of rows
            # This somehow fails on @blaz's side: Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so 
            # or libmkl_def.so. <- CONDA PROBLEM, for now using self.cosine
            #match = cosine(feats,  self.feat_db[i,  :])
            match = self.cosine(feats,  self.feat_db[i,  1:])
            person_id =  self.feat_db[i, 0]
            
            if (self.best_match < match):
                self.best_match = match
                self.best_id = person_id
        
        #print(self)
        
        return int(self.best_id) # TODO: verify if this is the right id
        
    def __str__(self):
        return "{} - Best ID / match: {} / {}".format(self.__class__.__name__,  self.best_id,  self.best_match) 
