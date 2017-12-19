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
        
    def generate_skin_mask(self,  img,  mask):
        # define the upper and lower boundaries of the HSV pixel
        # intensities to be considered 'skin'
        # B, G, R
        lower = np.array([0, 10, 20], dtype = "uint8")
        upper = np.array([200, 255, 255], dtype = "uint8")

        converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        #skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)
        #skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
        #skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)
        #skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)

        #skinMask = cv2.erode(skinMask, kernel, iterations = 1)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
       #
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(img, img, mask = skinMask)
        mask = cv2.bitwise_and(mask, mask, mask = skinMask)

        return skin, mask

    def replace_v2(self,  src_img,  src_roi,  gen_img,  _debug=False):

        if _debug:
            cv2.imshow('Generated before warp',  cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))
        
        x, y, w, h = src_roi
        m = min(w, h)

        if _debug:
            cv2.imshow('Mask before warp',  mask)

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

        src_face = src_img[y:y + h, x:x + w]
        hull=cv2.convexHull(src_pts)
        mask=np.zeros(src_face.shape[:2]+(1,),dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 1)

        # face_hist = cv2.calcHist([src_face//32], [0,1,2], mask, [8,8,8], None)
        # bp=cv2.calcBackProject([src_face//32],[0,1,2], face_hist, [0,8,0,8,0,8], 1)
        # bp-=bp.min()
        # mask=(bp>bp.max()*0.25).astype(np.uint8)

        # kernel = np.ones((4,4),np.uint8)
        # mask=cv2.erode(mask,kernel)
        if _debug:
            cv2.imshow('face', src_face)
            cv2.imshow('mask', mask*255)
            cv2.waitKey(0)

       # print("SRC: {}".format(src_pts))
       # print("GEN: {}".format(gen_pts)) 
        
        self.homography, status = cv2.findHomography(gen_pts, src_pts)
        
       # print("S: {}".format(status)) # homography matrix        
       # print("H: {}".format(h)) # homography matrix
        
        gen_img_warp = cv2.warpPerspective(gen_img, h, size, borderMode=cv2.BORDER_REPLICATE)

        gen_img_warp = self.equalize_colors(src_face, gen_img_warp,mask)
        
        if _debug:
            x, y, w, h = src_roi
        #    cv2.namedWindow("Replacer")
            cv2.rectangle(gen_img_warp,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.imshow('Warped gen face', cv2.cvtColor(gen_img_warp, cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
            #cv2.destroyAllWindows()

        return self.replace(src_img,  src_roi,  gen_img_warp, mask, _debug)

    def equalize_colors(self, src, gen, mask=None,_debug=False):
        if mask is None:
            mask=np.zeros(gen.shape[:2]+(1,),dtype=np.uint8)
            # pts = cv2.ellipse2Poly((mask.shape[0] // 2, mask.shape[1] // 2), (mask.shape[0] // 2, mask.shape[1] // 2), 0, 0, 360, 1)
            # cv2.fillPoly(mask, [pts], 1)
            cv2.ellipse(mask,(mask.shape[0] // 2, mask.shape[1] // 2), (mask.shape[0] // 3, mask.shape[1] // 3),0,0,360,1,3)
        res=np.empty_like(gen)
        for channel in range(3):
            src_hist=cv2.calcHist([src],[channel],mask,[256],None)
            gen_hist=cv2.calcHist([gen],[channel],mask,[256],None)
            src_cum=np.cumsum(src_hist)
            gen_cum=np.cumsum(gen_hist)
            src_cum_norm=(src_cum-src_cum.min())*255/(src_cum.max()-src_cum.min())
            gen_cum_norm=(gen_cum-gen_cum.min())*255/(gen_cum.max()-gen_cum.min())
            inv_src=np.zeros_like(src_cum_norm)
            prev=-1
            prevN=0
            for n,i in enumerate(src_cum_norm):
                i=int(i)
                inv_src[i]=n
                inv_src[prev:i]=prevN+np.arange(i-prev)/(i-prev)*(n-prevN)
                # for j in range(prev+1,i):
                #     inv_src[j]=prevN+()
                prev=i
                prevN=n
            #LUT=(gen_cum-gen_cum.min())/(gen_cum.max()-gen_cum.min())*(src.max()-src.min())+src.min()
            LUT=inv_src[gen_cum_norm.astype(np.int32)]
            res[:,:,channel]=LUT[gen[:,:,channel]]
        if _debug:
            cv2.imshow("generated face",gen)
            cv2.imshow("src face",src)
            cv2.imshow("color corrected face",res)
            cv2.waitKey(0)
        return res

    def replace(self, src_img, src_roi, gen_img, mask=None, _debug=False):
        x, y, w, h = src_roi
        # TODO: properly center the mask on larger axis
        m = max(w, h)
        if mask is None:
            mask = self.generateGaussianMask(m)
        else:
            mask=np.squeeze(mask).astype(np.float32)
            mask=cv2.GaussianBlur(mask,(9,9),3)
            # cv2.imshow("maskB",mask)
            # cv2.waitKey(0)
        
        alt_img = copy.copy(src_img)
        
        # blend generated image with Gaussian mask
        try:
            alt_img[y:y+h, x:x+w, 0] = alt_img[y:y+h, x:x+w, 0] * (1-mask) + mask * gen_img[:, :, 0] #*255.0
            alt_img[y:y+h, x:x+w, 1] = alt_img[y:y+h, x:x+w, 1] * (1-mask) + mask * gen_img[:, :, 1]  #*255.0
            alt_img[y:y+h, x:x+w, 2] = alt_img[y:y+h, x:x+w, 2] * (1-mask) + mask * gen_img[:, :, 2]  #*255.0
        except:
            import sys

            print("ERROR: Replacer - replace - unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
            print("ERROR: {}, {}, {}".format(gen_img[:, :, 0].shape, alt_img[y:y+h, x:x+w, 0].shape, mask.shape))
            raise
       # alt_img[y:y+m, x:x+m, 0] = (1-warp_mask) * alt_img[y:y+m, x:x+m, 0] + (warp_mask) * (alt_img[y:y+m, x:x+m, 0] * (1-mask) + mask * gen_img[:, :, 0]) #*255.0
       # alt_img[y:y+m, x:x+m, 1] = (1-warp_mask) * alt_img[y:y+m, x:x+m, 1] + (warp_mask) * (alt_img[y:y+m, x:x+m, 1] * (1-mask) + mask * gen_img[:, :, 1])  #*255.0
       # alt_img[y:y+m, x:x+m, 2] = (1-warp_mask) * alt_img[y:y+m, x:x+m, 2] + (warp_mask) * (alt_img[y:y+m, x:x+m, 2] * (1-mask) + mask * gen_img[:, :, 2])  #*255.0
        
        # hard-edge replacement
        # alt_img[y:y+m, x:x+m, 0] = gen_img[:, :, 0] #*255.0
        # alt_img[y:y+m, x:x+m, 1] = gen_img[:, :, 1]  #*255.0
        # alt_img[y:y+m, x:x+m, 2] = gen_img[:, :, 2]  #*255.0
        
        if _debug:
        #    cv2.namedWindow("Replacer")
           # cv2.rectangle(alt_img,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.imshow('Replaced face', cv2.cvtColor(alt_img, cv2.COLOR_RGB2BGR))
            cv2.imshow('Kernel', mask)
           # cv2.waitKey(200)
            #cv2.destroyAllWindows()
            #print(self)

        return alt_img
        
    # Origin: https://gist.github.com/andrewgiessel/4635563
    def generateGaussianMask(self, size, sigma=None, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is stddev (effective radius).
        """
        
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2 # integer division (Py.3 uses //)
        else:
            x0 = center[0]
            y0 = center[1]
        
        sigma = size / 5.0 # before size / 3.0 when 2 was not included in Gaussian
        return  np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma)**2)

    def generateEpanechnikMask(self,  size, width):
        # TODO
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
        
        pass

    def warpGaussianMask(self,  mask,  gen_img):
        mask[gen_img[:,  :, 0] == 0] = 0
        return mask

    def generateWarpMask(self, gen_img, _debug=False):
        warp_mask = copy.copy(gen_img)[:,  :,  0];
        warp_mask[warp_mask>0]=1
        warp_mask = warp_mask.astype(float)
        return warp_mask

    def __str__(self):
        return "{}".format(self.__class__.__name__) 
