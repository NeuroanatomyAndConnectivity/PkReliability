#!/bin/bash 
for i in 2 4 6 8 10;do  

sbatch send_jobCorrSLURM.sh ${i}

done
