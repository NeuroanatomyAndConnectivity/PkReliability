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

def checkPresence(thrGrad,hemi):
    
    if hemi =='L':
        g=thrGrad[0]
        WS=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/LWS.28.max.label.gii').darrays[0].data
    elif hemi =='R':
        g=thrGrad[1]
        WS=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/RWS.28.max.label.gii').darrays[0].data
#     print(hemi)
    del thrGrad
    
    mpar=np.where(WS==7)[0]
    mtmp=np.where(WS==8)[0]
    OP=np.where(WS==9)[0]
    
    mpar=np.intersect1d(mpar,g).shape[0] >0
    mtmp=np.intersect1d(mtmp,g).shape[0] >0
    OP=np.intersect1d(OP,g).shape[0] <1
    
    if mpar ==True and mtmp==True and OP==True:
#         print('do the thing')
        return 1
    elif mpar ==True and mtmp==False and OP==True:
#         print('we lost the medial temporal')
        return 0
        
    elif mpar ==True and mtmp==True and OP==False:
#         print('shit... occipital pole included')
        return 0
    
    elif mpar ==True and mtmp==False and OP==False:
#         print('we lost medial temporal but have occipital')
        return 0
    else:
        return 0

def optimThresh(ID):
    subj=hcp_subj(ID,4)
    L=[]
    R=[]
    thrs=list(range(60,99))
    for i in thrs:
        g=subj.subj.extract_topX(subj.Lgrad,subj.Rgrad,i)
        L.append(checkPresence(g,'L'))
        R.append(checkPresence(g,'R'))
    LRsum=np.sum(np.vstack([L,R]),axis=0)
    
    ideals=np.where(LRsum==2)[0]
    if ideals.shape[0]==0:
        return 0
    else:
        ideal=ideals[-1]
        return thrs[ideal]

def TopToCort(subj):
    Lsrf=(subj.Lcoords,subj.Lfaces)
    Rsrf=(subj.Rcoords,subj.Rfaces)

    thr=optimThresh(subj.subj)
    print(f'optimal threshold for {subj.subj} is {thr}')
    L10,R10=subj.extract_topX(subj.Lgrad,subj.Rgrad,thr)
    dL=surfdist.analysis.dist_calc(Lsrf,subj.Lfill,L10)
    dR=surfdist.analysis.dist_calc(Rsrf,subj.Rfill,R10)
    return dL,dR
    
Ltop,Rtop=TopToCort(subj)
np.save(f'{out}/L.topIndiThr',Ltop)
np.save(f'{out}/R.topIndiThr',Rtop)
