#!/usr/bin/env python

from hcp_class import hcp_subj
import numpy as np
import gdist
import nibabel as nib
import pickle
import sys
import os

subj=sys.argv[1]
spin=sys.argv[2]

out=f'Dist2SensoryBorder/{subj}/'
os.makedirs(out,exist_ok=True)
os.makedirs(f'{out}/spinBatches',exist_ok=True)

subj=hcp_subj(subj,4)

def get_spinsPickle(file):
    with open(file,'rb') as f:
        sp=pickle.load(f)
        return sp.spin_lh_,sp.spin_rh_

def gradDefROIs(subj):
    L10,R10=subj.extract_topX(subj.Lgrad,subj.Rgrad,90)
    
    Lws=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/LWS.28.max.label.gii').darrays[0].data
    Lfront=np.intersect1d(np.where(Lws==1)[0],L10)
    Lpar=np.intersect1d(np.where(Lws==2)[0],L10)
    Ltmp=np.intersect1d(np.where(Lws==5)[0],L10)
    Lmpar=np.intersect1d(np.where(Lws==7)[0],L10)
    Lrois=[Lfront,Lpar,Ltmp,Lmpar]
    
    Rws=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/RWS.28.max.label.gii').darrays[0].data
    
    Rfront=np.intersect1d(np.where(Rws==1)[0],R10)
    Rpar=np.intersect1d(np.where(Rws==2)[0],R10)
    Rtmp=np.intersect1d(np.where(Rws==5)[0],R10)
    Rmpar=np.intersect1d(np.where(Rws==7)[0],R10)
    Rrois=[Rfront,Rpar,Rtmp,Rmpar]
    return Lrois,Rrois

def rois2cort(subj,spin==False):
    L10,R10=gradDefROIs(subj)
    if spin==False:
        pass
    else:
        Lspin,Rspin=get_spinsPickle(spin)
        for i in range(len(Lspin)):
            sp=Lspin[i]
            for j in range(len(L10)):
                L10[j]=sp[L10[j]]
        for i in range(len(Lspin)):
            sp=Rspin[i]
            for j in range(len(R10)):
                R10[j]=sp[R10[j]]
    
    Lsrf=(subj.Lcoords,subj.Lfaces)
    Ldists=[]
    for i in L10:
        Ldists.append(surfdist.analysis.dist_calc(Lsrf,subj.Lfill,i))
        
    Rsrf=(subj.Rcoords,subj.Rfaces)
    Rdists=[]
    for j in R10:
        Rdists.append(surfdist.analysis.dist_calc(Rsrf,subj.Rfill,j))
    Ldists=np.vstack(Ldists)
    Rdists=np.vstack(Rdists)
    return Ldists,Rdists
  
##### get the distances from each ROI 
L,R=rois2cort(subj)

np.save(L,f'{out}/L.Peaks2cort',spin=False)
np.save(R,f'{out}/R.Peaks2cort',spin=False)
