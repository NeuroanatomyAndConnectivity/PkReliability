#!/bin/bash 
#SBATCH --job-name=PermutedDistPk
#SBATCH -o ./logs/PermutedDistPk-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array=1-912:1
#SBATCH --requeue

SUBJECT_LIST=./subjectsWithParietalPeak.txt


module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/postHoc-permutations-skl/bin/activate


echo the job id is $SLURM_ARRAY_JOB_ID
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "permuting distance measures on spun surface for $SUBJECT"

for i in `cat spinFiles.txt`;do 
    echo python3 peaks2cortex.py  $SUBJECT $i
    python3 -u peaks2cortex.py $SUBJECT $i
done

