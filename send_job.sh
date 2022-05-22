#!/bin/bash

#### load python environment 
module load Python/3.8.6-GCCcore-10.2.0
source /well/margulies/users/mnk884/python/pkReliability-skylake/bin/activate

##### load workbench 
module load ConnectomeWorkbench/1.4.2-rh_linux64

### job parameters

#$ -P PkRel
#$ -cwd
#$ -q short.qc
#$ -j y
#$ -o ./logs
#$ -pe shmem 9

SUBJECT_LIST=./subjectList.txt
nsubjs=`wc -l < $SUBJECT_LIST`

### each subject forms one job of the array job

#$ -t 1-$nsubjs


####get file name 
FILENAME=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)

echo "Processing subject $FILENAME"

# Load a recent python module

# run pk reliability for subject 
python pkReliability.py --subj $FILENAME  --odir /well/margulies/projects/pkReliability 
