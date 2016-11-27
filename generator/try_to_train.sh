#!/bin/bash

# dumb way of training to achieve better results (standalone runs often exit randomly)

MODEL="FaceGen.RaFD.model.d5.nadam.h5" #"FaceGen.RaFD.model.d5.adam.h5"
#JOBS=("interpolate" "random" "single" "drunk")
STEPS=50
DB_DIR="rafd2-frontal/"

# activate conda environment with keras & tensorflow
source activate python35

for i in $(seq 1 $STEPS)
do
    echo "TRYOUT $i / $STEPS"
    python faces.py train $DB_DIR -m output/${MODEL} -b 8 -opt nadam
done

# source deactivate python35