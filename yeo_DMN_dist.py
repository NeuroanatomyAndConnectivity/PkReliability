#! /usr/bin/env python
from hcp_class import hcp_subj
import numpy as np
import gdist
import surfdist.analysis
import surfdist.utils
import surfdist
import nibabel as nib
import pickle
import sys
import os

subj=sys.argv[1]
out=f'/well/margulies/projects/pkReliability/Dist2SensoryBorder/{subj}/'
os.makedirs(out,exist_ok=True)

subj=hcp_subj(subj,4)
print(subj.subj)    

Lsrf=(subj.Lcoords,subj.Lfaces)
Rsrf=(subj.Rcoords,subj.Rfaces)

Lyeo=np.load('./32k_fs_LR/yeo/yeo.L.DMN.npy')
Lyeo=np.where(Lyeo==1)[0]
Lyeo=surfdist.analysis.dist_calc(Lsrf,subj.Lfill,Lyeo)
np.save(f'{out}/L.yeo',Lyeo)


Ryeo=np.load('./32k_fs_LR/yeo/yeo.R.DMN.npy')
Ryeo=np.where(Ryeo==1)[0]
Ryeo=surfdist.analysis.dist_calc(Rsrf,subj.Rfill,Ryeo)
np.save(f'{out}/R.yeo',Ryeo)

