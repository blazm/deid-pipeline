#!/bin/bash

# dumb way of training to achieve better results (standalone runs often exit randomly)

MODEL="FaceGen.RaFD.model.d5.adam.h5" #"FaceGen.RaFD.model.d5.adam.h5"
#JOBS=("interpolate" "random" "single" "drunk")
DB_DIR="rafd2-frontal/"

# @refik we dont need a for loop here. Just increase the epoch number.
# with -d 6 model create bigger images.

echo "TRAINING STARTED..."
python faces.py train $DB_DIR -d 6 -b 16 -opt adam -e 500

# source deactivate python35