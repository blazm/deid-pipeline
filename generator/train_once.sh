#!/bin/bash

MODEL="FaceGen.RaFD.model.b16.e500.d6.adam.Ti.h5"
#JOBS=("interpolate" "random" "single" "drunk")
STEPS=50
DB_DIR="../DB/rafd2-frontal/" # To prevent Dropbox from syncing +3GB

existing=$1 # if param1 exists, then we use existing model, else we train a new one

batch_size=16
optimizer="adam"
epochs=500
deconv_layers=6

#-m models/${MODEL} 

if [ -n "$existing" ]; then
    echo "Using existing model: ${MODEL}"
    python faces.py train $DB_DIR -m models/${MODEL} -b ${batch_size} -opt ${optimizer} -e ${epochs} -d ${deconv_layers}
else
    echo "Learning new model: ${MODEL}"
    python faces.py train $DB_DIR -o models/${MODEL} -b ${batch_size} -opt ${optimizer} -e ${epochs} -d ${deconv_layers}
fi
