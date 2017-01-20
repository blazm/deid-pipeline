
#from scipy.spatial.distance import cosine
import numpy as np

from collections import defaultdict

class Matcher:
    
    def __init__(self,  feat_db):
        # TODO: load DB?
        self.feat_db = feat_db
        self.best_match = 0
        self.best_id = None
        
        self.matches = defaultdict() # match cache
        
    def cosine(self, a, b):
        return np.dot(a,b.T)/np.linalg.norm(a)/np.linalg.norm(b)
        
    def match_n(self, feats):
        pass

    def match(self, feats, max_person_id, selected_person_id=None):  
        
        # return cached best id
        if selected_person_id:
            if selected_person_id in self.matches:
                return self.matches[selected_person_id]
        
        self.best_match = 0
        self.best_id = None
        for i in range(self.feat_db.shape[0]): # number of rows
            # This somehow fails on @blaz's side: Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so 
            # or libmkl_def.so. <- CONDA PROBLEM, for now using self.cosine
            #match = cosine(feats,  self.feat_db[i,  :])
            match = self.cosine(feats,  self.feat_db[i, 1:])
            person_id = self.feat_db[i, 0]
            
            if (self.best_match < match and person_id < max_person_id): # select best person id, which is still small enough (trained GNN model limitation)
                self.best_match = match
                self.best_id = person_id
                
        # cache best id if not cached yet
        if selected_person_id:
            if selected_person_id not in self.matches:
                self.matches[selected_person_id] = int(self.best_id)
                
        return int(self.best_id) # TODO: verify if this is the right id
        
    def __str__(self):
        return "{} - Best ID / match: {} / {}".format(self.__class__.__name__,  self.best_id,  self.best_match) 
