#!/bin/bash

### job parameters
#SBATCH --job-name=gradGen
#SBATCH -o ./logs/gradGenJob-%j.out
#SBATCH -p short
#SBATCH --cpus-per-task=10
#SBATCH --array=1-5:1
#SBATCH --requeue
SUBJECT_LIST=./test.txt
smooth_kernel=$1

#####load workbench
module load ConnectomeWorkbench/1.4.2-rh_linux64
#### load python
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/corrmats-skylake/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
### each subject forms one job of the array job

echo "smoothing kernel is" ${smooth_kernel}

####get file name 
echo the job id is $SLURM_ARRAY_JOB_ID
FILENAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo echo $SLURM_ARRAY_JOB_ID
echo "Processing subject $FILENAME"

# Load a recent python module

# run duffusion map embedding for subject 
#python GradDistCorrTesting.py --subj $FILENAME  --odir /well/margulies/projects/pkReliability

python -u GradDistCorrFullHCP.py --subj $FILENAME  --odir /well/margulies/projects/pkReliability --kernel ${smooth_kernel}

