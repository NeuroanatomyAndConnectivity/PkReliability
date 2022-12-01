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


def MultiThr70to99(subj):
    Lsrf=(subj.Lcoords,subj.Lfaces)
    Rsrf=(subj.Rcoords,subj.Rfaces)
    dL=[]
    dR=[]
    for thr in range(70,99):
      L10,R10=subj.extract_topX(subj.Lgrad,subj.Rgrad,thr)
      dL.append(surfdist.analysis.dist_calc(Lsrf,subj.Lfill,L10))
      dR.append(surfdist.analysis.dist_calc(Rsrf,subj.Rfill,R10))
    return np.vstack(dL),np.vstack(dR)
    
Ltop,Rtop=MultiThr70to99(subj)
np.save(f'{out}/L.70to99Thr',Ltop)
np.save(f'{out}/R.70to99Thr',Rtop)
