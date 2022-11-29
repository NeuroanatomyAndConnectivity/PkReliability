#!/bin/bash

module load ConnectomeWorkbench/1.4.2-rh_linux64
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/postHoc-permutations-skl/bin/activate


### job parameters
#$ -N Watershed
#$ -o logs/
#$ -e logs/
#$ -j y
#$ -cwd
#$ -q short.qc
#$ -pe shmem 3
#$ -t 1-912


SUBJECT_LIST=./subjectsWithParietalPeak.txt

SUBJECT=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)
echo python3 peaks2cortex.py  $SUBJECT $i
python3 -u IndividualWatershed.py $SUBJECT
