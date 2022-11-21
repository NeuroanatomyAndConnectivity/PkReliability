#!/bin/bash


module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/postHoc-permutations-skl/bin/activate


### job parameters
#$ -N Permute
#$ -o logs/
#$ -e logs/
#$ -j y
#$ -cwd
#$ -q short.qc
#$ -pe shmem 1
#$ -t 1-912


SUBJECT_LIST=./subjectsWithParietalPeak.txt

params=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)

for i in `cat spinFiles.txt`;do 
    echo python3 peaks2cortex.py  $SUBJECT $i
    python3 -u peaks2cortex.py $SUBJECT $i
done
