#!/bin/bash 

for i in `cat subjectIDs_3T.txt`;do  

outerPath=/well/win-hcp/HCP-YA/subjectsAll/${i}/MNINonLinear/Results/
	if [ -d ${outerPath}/rfMRI_REST1_RL ] && [ -d ${outerPath}/rfMRI_REST1_LR ] && [ -d ${outerPath}/rfMRI_REST2_RL ] && [ -d ${outerPath}/rfMRI_REST2_LR ];then 
		echo ${i} >>SubjectsCompleteData.txt
	else 
		echo ${i} >>SubjectsMissingRuns.txt
	fi 

done
