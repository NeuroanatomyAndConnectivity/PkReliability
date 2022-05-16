#!/bin/bash

#$ -cwd
#$ -q short.qc
#$ -j y
#$ -o ./logs


SUBJECT_LIST=./subjectList.txt
nsubjs=`wc -l < $SUBJECT_LIST`

#$ -t 1-$nsubjs



FILENAME=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)

echo "Processing subject $FILENAME"

# Load a recent python module
module load Python/3.7.0-foss-2018b

# run pk reliability for subject 
python pkReliability.py --subj $FILENAME  100206 --odir /well/margulies/projects/pkReliability 
