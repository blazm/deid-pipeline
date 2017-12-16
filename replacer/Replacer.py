import numpy as np
import copy
import cv2
from numpy import float32,  uint32

import dlib
#use keypoints to align faces!

class Replacer:
    
    def __init__(self):
        self.mask = None
        predictor_path = "./replacer/shape_predictor_68_face_landmarks.dat"
        #detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
    def shape_to_nparray(self, shape,  displacement=(0, 0)):
        """
        Reshapes Shape from dlib predictor to numpy array
        Args:
            shape (dlib Shape object): input shape points
            displacement (x, y): tuple with displacement components, which are subtracted

        Returns: numpy array consisting of shape points
        """
        dx, dy = displacement
        np_arr = []
        for i in range(0,  shape.num_parts):
            np_arr.append((shape.part(i).x - dx,  shape.part(i).y - dy))
        return np.array(np_arr)
        
        
    def replace_v2(self,  src_img,  src_roi,  gen_img,  _debug=False):

        
        x, y, w, h = src_roi
        size = (w, h)
        # convert to use dimensions in dlib.rectangle
        ix = uint32(x).item()
        iy = uint32(y).item()
        iw = uint32(w).item()
        ih = uint32(h).item()
        
        #src_detection = dlib.rectangle(left=x*1.0, top=y*1.0, right=x+w*1.0, bottom=y+h*1.0)
        #gen_detection = dlib.rectangle(left=0.0, top=0.0, right=w*1.0, bottom=h*1.0)
        src_detection = dlib.rectangle(ix, iy, ix+iw, iy+ih)
        gen_detection = dlib.rectangle(0, 0, iw, ih)
        
        src_shape = self.predictor(src_img, src_detection)
        gen_shape = self.predictor(gen_img, gen_detection)
        
        displacement = (x, y)
        src_pts = self.shape_to_nparray(src_shape,  displacement)
        gen_pts = self.shape_to_nparray(gen_shape)

       # print("SRC: {}".format(src_pts)) 
       # print("GEN: {}".format(gen_pts)) 
        
        h, status = cv2.findHomography(gen_pts, src_pts)
        
       # print("S: {}".format(status)) # homography matrix        
       # print("H: {}".format(h)) # homography matrix
        
        gen_img_warp = cv2.warpPerspective(gen_img, h, size)
        
        if _debug:
            x, y, w, h = src_roi
        #    cv2.namedWindow("Replacer")
            cv2.rectangle(gen_img_warp,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.imshow('Warped gen face', cv2.cvtColor(gen_img_warp, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
        return self.replace(src_img,  src_roi,  gen_img_warp, _debug)
  
    def replace(self, src_img, src_roi, gen_img,  _debug=False):  
        x, y, w, h = src_roi
        # TODO: properly center the mask on larger axis
        m = max(w, h)
        mask = self.generateGaussianMask(m)
        
        alt_img = copy.copy(src_img)
        
        # blend generated image with Gaussian mask
        alt_img[y:y+m, x:x+m, 0] = alt_img[y:y+m, x:x+m, 0] * (1-mask) + mask * gen_img[:, :, 0] #*255.0
        alt_img[y:y+m, x:x+m, 1] = alt_img[y:y+m, x:x+m, 1] * (1-mask) + mask * gen_img[:, :, 1]  #*255.0
        alt_img[y:y+m, x:x+m, 2] = alt_img[y:y+m, x:x+m, 2] * (1-mask) + mask * gen_img[:, :, 2]  #*255.0
        
        # hard-edge replacement
#        alt_img[y:y+m, x:x+m, 0] = gen_img[:, :, 0] #*255.0
#        alt_img[y:y+m, x:x+m, 1] = gen_img[:, :, 1]  #*255.0
#        alt_img[y:y+m, x:x+m, 2] = gen_img[:, :, 2]  #*255.0
        
        if _debug:
        #    cv2.namedWindow("Replacer")
            cv2.rectangle(alt_img,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.imshow('Replaced face', cv2.cvtColor(alt_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
        print(self)
        return alt_img
        
    # Origin: https://gist.github.com/andrewgiessel/4635563
    def generateGaussianMask(self, size, fwhm = 3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2 # integer division (Py.3 uses //)
        else:
            x0 = center[0]
            y0 = center[1]
        
        fwhm = size *0.85 #((size//2) - 1) + 0.8
    
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        
    def __str__(self):
        return "{}".format(self.__class__.__name__) 
