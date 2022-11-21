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
out=f'Dist2SensoryBorder/{subj}/'
os.makedirs(out,exist_ok=True)
os.makedirs(f'{out}/spinBatches',exist_ok=True)

subj=hcp_subj(subj,4)

spinned=sys.argv[2]


def load_pickle(file):
    with open(file,'rb') as data:
        return pickle.load(data)

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

def rois2cort(subj,spin=False):
    Lsrf=(subj.Lcoords,subj.Lfaces)
    Rsrf=(subj.Rcoords,subj.Rfaces)
    L10,R10=gradDefROIs(subj)
    if spin !=False:
        print('spinning')
        sp=load_pickle(spin)
        Ldists=[]
        for lh in sp.spin_lh_:
            Lrois=[]
            for roi in L10:
                ##### cortex mask needs to be sorted to work with surfdist
                d=surfdist.analysis.dist_calc(Lsrf,np.sort(lh[subj.Lfill]),lh[roi])
                d[np.where(np.isfinite(d)!=True)[0]]=0
                Lrois.append(d)
            Ldists.append(np.vstack(Lrois))
        ### right hemispehre
        Rdists=[]
        for rh in sp.spin_rh_:
            Rrois=[]
            for roi in R10:
                d=surfdist.analysis.dist_calc(Rsrf,np.sort(rh[subj.Rfill]),rh[roi])
                d[np.where(np.isfinite(d)!=True)[0]]=0
                Rrois.append(d)
            Rdists.append(np.vstack(Rrois))
        return Ldists,Rdists
    
    else:
        
        
        Ldists=[]
        for roi in L10:
            Ldists.append(surfdist.analysis.dist_calc(Lsrf,subj.Lfill,roi))
        
        Rdists=[]
        for roi in R10:
            Rdists.append(surfdist.analysis.dist_calc(Rsrf,subj.Rfill,roi))
        Ldists=np.vstack(Ldists)
        Rdists=np.vstack(Rdists)
        return Ldists,Rdists

print('measuring true value')
L,R=rois2cort(subj)

print('measuring permuted values')
L,R=rois2cort(subj,spin=spinned)

print('task failed succesfully')
