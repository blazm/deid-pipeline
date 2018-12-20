#SRC: http://dlib.net/face_landmark_detection.py.html
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 


#TO install dlib: conda install --name python35 -c menpo dlib=18.18
# OR:  conda install --name python35 -c conda-forge dlib=19.0
# imports were a bit modified since only dlib was used (version from menpo does not have image_window function)

import sys
import dlib
import scipy.misc
#from skimage import io
import cv2
import numpy as np

img = scipy.misc.imread('./in/people2.jpg')

detector = dlib.get_frontal_face_detector()
predictor_path = "./replacer/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

#for f in sys.argv[1:]:
#    print("Processing file: {}".format(f))
#    img = io.imread(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
shape = None
    
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print(d.__class__)
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))

    
    prev_shape = shape
    shape = predictor(img, d)
    print("Shape: LEN: {} Part 0: {}, Part 1: {} ...".format(shape.num_parts, shape.part(0), shape.part(1)))
    # Draw the face landmarks on the screen.
#    win.add_overlay(shape)

def shape_to_nparray(shape):
    """
    Reshapes Shape from dlib predictor to numpy array
    Args:
        shape (dlib Shape object): input shape points

    Returns: numpy array consisting of shape points

    """
    np_arr = []
    for i in range(0,  shape.num_parts):
        np_arr.append((shape.part(i).x,  shape.part(i).y))
    return np.array(np_arr)
    
shape_to_nparray(shape)

# TODO: homography estimation
# SRC: http://www.learnopencv.com/homography-examples-using-opencv-python-c/
'''
pts_src and pts_dst are numpy arrays of points
in source and destination images. We need at least 
4 corresponding points. 
'''

pts_src = shape_to_nparray(shape)
pts_dst = shape_to_nparray(prev_shape)

print(pts_src)

h, status = cv2.findHomography(pts_src, pts_dst)
 
print("H: {}".format(h)) # homography matrix
''' 
The calculated homography can be used to warp 
the source image to destination. Size is the 
size (width,height) of im_dst
'''
# TODO: test how this works
#im_dst = cv2.warpPerspective(im_src, h, size)

#win.clear_overlay()
#win.set_image(img)
#win.add_overlay(dets)
#dlib.hit_enter_to_continue()


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
#if (len(sys.argv[1:]) > 0):
#    img = io.imread(sys.argv[1])
#    dets, scores, idx = detector.run(img, 1, -1)
#    for i, d in enumerate(dets):
#        print("Detection {}, score: {}, face_type:{}".format(
#            d, scores[i], idx[i]))
