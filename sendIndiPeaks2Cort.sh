#!/bin/bash


module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/postHoc-permutations-skl/bin/activate


### job parameters
#$ -N IndiDist
#$ -o logs/
#$ -e logs/
#$ -j y
#$ -cwd
#$ -q short.qc
#$ -pe shmem 1
#$ -t 1-776


SUBJECT_LIST=./text_files/IndividualThresholdSubjects.txt

SUBJECT=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)
echo python3 -u IndiSpecificThrDist.py  $SUBJECT
python3 -u IndiSpecificThrDist.py $SUBJECT 
