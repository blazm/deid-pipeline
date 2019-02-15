from os.path import join, split, realpath
import numpy as np
import copy
import cv2
from numpy import float32,  uint32

import dlib
#use keypoints to align faces!

from scipy.misc import imresize

# config from: https://github.com/matthewearl/faceswap/blob/master/faceswap.py
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                            RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + FACE_POINTS + JAW_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS + FACE_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

class Replacer:
    
    def __init__(self):
        self.mask = None
        
        predictor_path = "./replacer/shape_predictor_68_face_landmarks.dat" # join(split(realpath(__file__))[0], "shape_predictor_68_face_landmarks.dat")
        #detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        self.src_face_landmarks = None
        self.gen_img = None
        self.src_mask = None

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

    # source: https://matthewearl.github.io/2015/07/28/switching-eds-with-python/




    def get_landmarks(self, im):
        rects = detector(im, 1)
        
        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R,
                                        c2.T - (s2 / s1) * R * c1.T)),
                            np.matrix([0., 0., 1.])])

    def warp_im(self, im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                    M[:2],
                    (dshape[1], dshape[0]),
                    dst=output_im,
                    borderMode=cv2.BORDER_TRANSPARENT,
                    flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                                np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        #im2_blur += 128 * (im2_blur <= 1.0)
        im2_blur = np.add(im2_blur, 128 * (im2_blur <= 1.0), out=im2_blur, casting="unsafe")

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                    im2_blur.astype(np.float64))




    def draw_convex_hull(self, im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(self, im, landmarks):
        im = np.zeros(im.shape[:2], dtype=np.float64)

        for group in OVERLAY_POINTS:
            self.draw_convex_hull(im,
                            landmarks[group],
                            color=1)

        im = np.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im



    
    def replace_v3(self,  src_img,  src_roi,  gen_img,  _debug=False):

        #gen_img=cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        
        #if _debug:
        #    cv2.imshow('Generated before warp',  cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))
        x, y, w, h = src_roi
        self.gen_img = imresize(gen_img.copy(), (h, w, 3))

        m = min(w, h)

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

        landmarks2 = np.matrix([[p.x, p.y] for p in gen_shape.parts()])
        landmarks1 = np.matrix([[p.x - x, p.y - y] for p in src_shape.parts()])

        im2 = self.gen_img
        im1 = src_img[y:y+h, x:x+w, :] 

        M = self.transformation_from_points(landmarks1[ALIGN_POINTS],
                               landmarks2[ALIGN_POINTS])

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = np.max([self.get_face_mask(im1, landmarks1), warped_mask],
                                axis=0)
                                
        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

        #center = (int(h/2), int(w/2))
        #output_im = cv2.seamlessClone(im2, im1, warped_mask, center, cv2.NORMAL_CLONE)

        alt_img = src_img
        alt_img[y:y+h, x:x+w, :] = output_im

        return alt_img
    # -------------------------------------------------------------------------------

    

    def replace_v2(self,  src_img,  src_roi,  gen_img,  _debug=False):
        gen_img=cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        
        #if _debug:
        #    cv2.imshow('Generated before warp',  cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))
        x, y, w, h = src_roi
        self.gen_img = imresize(gen_img.copy(), (h, w, 3))

        m = min(w, h)

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
        src_face2 = None
        if _debug:
            self.src_face_landmarks=src_face.copy()
            self.src_face_landmarks = cv2.cvtColor(self.src_face_landmarks, cv2.COLOR_RGB2BGR)
            self.src_face_landmarks[np.minimum(src_face.shape[0] - 1, np.maximum(src_pts[:, 1], 0)),
                     np.minimum(src_face.shape[1] - 1, np.maximum(0,src_pts[:,0]))] = [0,255,0]
            

            #cv2.imshow("Source Face", cv2.cvtColor(src_face, cv2.COLOR_RGB2BGR))
            #cv2.moveWindow('Source Face', 0,30)
            
            #final = np.hstack((cv2.cvtColor(src_face2, cv2.COLOR_RGB2BGR), mask*255))
            #def x3(gray):
            #    return np.array([gray,gray,gray]).transpose((1, 2, 0))
            self.src_mask = np.repeat(mask*255, 3, axis=2)
            #print("SRC MASK: ", self.src_mask.shape)
            #cv2.imshow('Source Mask', mask*255)
            #cv2.moveWindow('Source Mask', 0,60)
            
       # print("SRC: {}".format(src_pts))
       # print("GEN: {}".format(gen_pts)) 
        
        self.homography, status = cv2.findHomography(gen_pts, src_pts)
        
       # print("S: {}".format(status)) # homography matrix        
       # print("H: {}".format(h)) # homography matrix
        
        gen_img_warp = cv2.warpPerspective(gen_img, self.homography, size, borderMode=cv2.BORDER_REPLICATE)

        gen_img_warp = self.equalize_colors(src_face, gen_img_warp, mask, True, (gen_img, src_face2))
        
        #gen_img_warp = cv2.cvtColor(gen_img_warp, cv2.COLOR_RGB2BGR)

        if _debug:
            x, y, w, h = src_roi
        #    cv2.namedWindow("Replacer")
            #cv2.rectangle(gen_img_warp,(x,y),(x+w,y+h),(0,255,0),1)
            #def x3(gray):
            #    return np.array([gray,gray,gray]).transpose((1, 2, 0))

            #final = np.hstack((src_face, self.src_face_landmarks, gen_img_warp, np.repeat(mask[:, :, 0]*255.0, 2)))
            #final = np.hstack((cv2.cvtColor(src_face2, cv2.COLOR_RGB2BGR), mask*255))
        

            #cv2.imshow('Source Face / Source Mask / Warped Face', final)

            #cv2.imshow('Warped gen face / Mask', final)
            #final = np.hstack((src, gen, res, gen_img_warp))

            #cv2.imshow("Source Face / Generated Face / Color Correction / Generated Warped", final)
            

            #cv2.imshow('Warped gen face', gen_img_warp)
            #cv2.moveWindow('Warped gen face', 40,30)
            
            #cv2.waitKey(1)
            #cv2.destroyAllWindows()
        gen_img_warp = cv2.cvtColor(gen_img_warp, cv2.COLOR_RGB2BGR)

        return self.replace(src_img,  src_roi,  gen_img_warp, mask, _debug)


    def equalize_colors(self, src, gen, mask=None,_debug=False, add_imgs=None):
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
        
        ##return res
        maskThreshold=150
        mask=cv2.cvtColor(gen,cv2.COLOR_RGB2GRAY)
        mask[mask<maskThreshold]=maskThreshold
        mask-=maskThreshold
        mask=(mask/(255-maskThreshold))[:,:,np.newaxis]

        #mask=np.squeeze(mask).astype(np.float32)
        #mask=cv2.GaussianBlur(mask,(9,9),3,borderType=cv2.BORDER_CONSTANT)

        final = None
        if _debug:

            def x3(gray):
                return np.array([gray,gray,gray]).transpose((1, 2, 0))

            src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            #mask = x3(mask)
            #print(src.shape, res.shape, mask.shape, gen.shape)

            #print(src.shape, self.gen_img.shape)

            final = np.hstack((src, self.src_face_landmarks, self.src_mask, self.gen_img, gen, res))
        
            
            cv2.imshow("Source Face / Landmarks / Generated Face / Warped Face / Color Correction", final)
            #cv2.moveWindow("Generated Face", 80,30)
            
            #cv2.imshow("Source Face", cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
            #cv2.moveWindow("Source Face", 120,30)

            #cv2.imshow("Color-corrected Face",cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            #cv2.moveWindow("Color-corrected Face", 160,30)
            #cv2.waitKey(1)

            #cv2.imshow("CC MASK",mask)

        return (res*(1-mask) + mask*gen.astype(np.float)/gen.mean((0,1))*src.mean((0,1))).astype(np.uint8)

    def replace(self, src_img, src_roi, gen_img, mask=None, _debug=False):
        x, y, w, h = src_roi
        # TODO: properly center the mask on larger axis
        m = max(w, h)
        if mask is None:
            mask = self.generateGaussianMask(m)
        else:
            mask=np.squeeze(mask).astype(np.float32)
            mask=cv2.GaussianBlur(mask,(9,9),3,borderType=cv2.BORDER_CONSTANT)
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
            cv2.imshow('Replaced Face(s)', cv2.cvtColor(alt_img, cv2.COLOR_RGB2BGR))
            #cv2.moveWindow('Replaced Face(s)', 680,200)

            cv2.imshow('Kernel', mask)
            #cv2.moveWindow('Kernel', 200,30)
            #cv2.waitKey(1)
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
 