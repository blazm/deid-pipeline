#!/bin/bash

# TODO: assemble model name from common params

MODEL="FaceGen.RaFD.model.d6.adam.h5" #"FaceGen.RaFD.model.d5.adam.h5"
#JOBS=("interpolate" "random" "single" "drunk") # all jobs
JOBS=("random") # only generate single image (ID is set up in params/single.yaml)
STAMP=`date '+%d_%m_%Y__%H_%M_%S'`;

# clean-up previous generations
#for job in "${JOBS[@]}"
#do
#done

# @refik I have changed the model input folder to output of training. It was models/ .
# To make it more practical, I added time stamp to the output folder name.
# generate new images
filename=generated-$STAMP.zip
for job in "${JOBS[@]}"
do
    # generate data
    python faces.py generate -m models/${MODEL} -o ../out/generated-${job}-${STAMP}/ -f params/${job}.yaml
    
    # zip generated data into one file
    #zip -ur $filename ../out/generated-${job}/
    
    # remove all generated dirs
    #rm -rf ..out/generated-${job}/
done



#python faces.py generate -m output/FaceGen.RaFD.model.d5.adam.h5 -o generated-interpolate/ -f params/interpolate.yaml 
