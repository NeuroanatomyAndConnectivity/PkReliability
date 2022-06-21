#!/bin/bash

#####load workbench 
module load ConnectomeWorkbench/1.4.2-rh_linux64
#### load python
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/pkReliability-skylake/bin/activate
### job parameters

#$ -P PkRel
#$ -cwd
#$ -q test.qc #short.qc
#$ -j y
#$ -o ./logs
#$ -pe shmem 1

SUBJECT_LIST=./subjectList.txt
nsubjs=`wc -l < $SUBJECT_LIST`

### each subject forms one job of the array job

#$ -t 1-$nsubjs


####get file name 
FILENAME=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)

echo "Processing subject $FILENAME"

# Load a recent python module

# run pk reliability for subject 
python GradDistCorr.py --subj $FILENAME  --odir /well/margulies/projects/pkReliability 