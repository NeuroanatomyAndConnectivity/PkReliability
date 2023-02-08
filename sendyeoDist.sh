#!/bin/bash 
#SBATCH --job-name=YeoDistDMN
#SBATCH -o ./logs/YeoDist-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array=1-912:1
#SBATCH --requeue

SUBJECT_LIST=./text_files/subjectsWithParietalPeak.txt


module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/measureDist-skylake/bin/activate


echo the job id is $SLURM_ARRAY_JOB_ID
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "measuring distance form YEO gorup DMN $SUBJECT"

python3 -u yeo_DMN_dist.py  $SUBJECT 
