#!/bin/bash 
#SBATCH --job-name=permute
#SBATCH -o ./logs/permute-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array=1-100:1
#SBATCH --requeue

SEED_LIST=./spins/seeds.txt



module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/postHoc-permutations-skl/bin/activate


echo the job id is $SLURM_ARRAY_JOB_ID
SEED=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SEED_LIST)
echo "spinning with random seed  $SEED"

python -u spinPermute.py $SEED

