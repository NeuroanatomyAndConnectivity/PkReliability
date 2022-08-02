#!/bin/bash 
for i in 2 4 6 8 10;do  

qsub send_jobCorr.sh ${i}

done
