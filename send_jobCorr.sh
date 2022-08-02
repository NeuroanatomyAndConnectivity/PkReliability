#!/bin/bash

#####load workbench 
module load ConnectomeWorkbench/1.4.2-rh_linux64
#### load python
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/corrmats-skylake/bin/activate
### job parameters
#$ -N gradGen
#$ -cwd
#$ -q short.qc
#$ -j y
#$ -o ./logs
#$ -pe shmem 10
#$ -t 1-20
SUBJECT_LIST=./subjectList.txt

### each subject forms one job of the array job



####get file name 
FILENAME=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)

echo "Processing subject $FILENAME"

# Load a recent python module

# run duffusion map embedding for subject 
#python GradDistCorrTesting.py --subj $FILENAME  --odir /well/margulies/projects/pkReliability
python -u GradDistCorr.py --subj $FILENAME  --odir /well/margulies/projects/pkReliability --kernel 4
