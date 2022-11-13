#!/usr/bin/env python

from hcp_class import hcp_subj
import numpy as np
import gdist
import nibabel as nib
import pickle
import sys
import os

subj=sys.argv[1]
out=f'Dist2SensoryBorder/{subj}/'
os.makedirs(out,exist_ok=True)
os.makedirs(f'{out}/spinBatches',exist_ok=True)

subj=hcp_subj(subj,4)

spin=sys.argv[2]

def get_spinsPickle(file):
    with open(file,'rb') as f:
        sp=pickle.load(f)
        return sp.spin_lh_,sp.spin_rh_
        
Lspin,Rspin=get_spinsPickle(spin)

Lrois=[np.load('Dist2SensoryBorder/LfRoi.npy'),np.load('Dist2SensoryBorder/LpRoi.npy')]
Rrois=[np.load('Dist2SensoryBorder/RfRoi.npy'),np.load('Dist2SensoryBorder/RpRoi.npy')]

def getNN(coords,faces,vertices):
    """Gets nearest neighbours of a single or iterable  vertices"""
    if type(vertices)==int:
        vertices=[vertices]
    mini_dict={}
    for i in vertices:
        a=np.unique(faces[np.where(faces==i)[0]])
        a=a[a!=i]
        mini_dict[i]=a
    return mini_dict 

def borders_betweenLabels(subj,hemi,label1verts,label2verts):
    if hemi =='L':
        coords=subj.Lcoords
        faces=subj.Lfaces
        l1=getNN(coords,faces,label1verts)
        l1=np.unique(list(l1.values()))
        l2=getNN(coords,faces,label2verts)
        l2=np.unique(list(l2.values()))
        return np.intersect1d(l1,l2)
    elif hemi =='R':
        coords=subj.Rcoords
        faces=subj.Rfaces
        l1=getNN(coords,faces,label1verts)
        l1=np.unique(list(l1.values()))
        l2=getNN(coords,faces,label2verts)
        l2=np.unique(list(l2.values()))
        return np.intersect1d(l1,l2)

def borderROI(subjClass):
    LpostCent=np.where(nib.load(subjClass.Laparc).darrays[0].data==28)[0]
    Lcentral=subjClass.LS1 
    Ltarg=borders_betweenLabels(subjClass,'L',Lcentral,LpostCent)
    print(f'left target is the border of the central sulcus and the post central gyrus with shape {Ltarg.shape}')
    #### do the right hemisphere now 
    RpostCent=np.where(nib.load(subjClass.Raparc).darrays[0].data==28)[0]
    Rcentral=subjClass.RS1 
    Rtarg=borders_betweenLabels(subjClass,'R',Rcentral,RpostCent)
    print(f'right target is the border of the central sulcus and the post central gyrus with shape {Rtarg.shape}')
    return Ltarg,Rtarg

def getDist2borderRatio(subj,roi1,roi2,border,hemi):
    """calculate the distance to a border region. first input is an instance of the hcp_subj class"""
    
    if hemi =='L':
        verts=subj.Lcoords.astype('float64')
        faces=subj.Lfaces.astype('int32')
    elif hemi=='R':
        verts=subj.Rcoords.astype('float64')
        faces=subj.Rfaces.astype('int32')
        
    roi1=roi1.astype('int32')
    roi2=roi2.astype('int32')
    
    border=border.astype('int32')
    
    dist1=gdist.compute_gdist(verts,faces,roi1,border)
    dist2=gdist.compute_gdist(verts,faces,roi2,border)
    
    ratio=dist1/(dist1+dist2)
    return ratio
def gradDefROIs(subj):
    L10,R10=subj.extract_topX(test.Lgrad,test.Rgrad,90)
    
    Lws=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/LWS.28.max.label.gii').darrays[0].data
    Lfront=np.where(Lws==1)[0]
    Lpar=np.where(Lws==2)[0]
    
    Lfront=np.intersect1d(Lfront,L10)
    Lpar=np.intersect1d(Lpar,L10)
    Lrois=[Lfront,Lpar]
    
    Rws=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/RWS.28.max.label.gii').darrays[0].data
    Rfront=np.where(Rws==1)[0]
    Rpar=np.where(Rws==2)[0]
    
    Rfront=np.intersect1d(Rfront,R10)
    Rpar=np.intersect1d(Rpar,R10)
    Rrois=[Rfront,Rpar]
    return Lrois,Rrois
    

#### save the canonical measurements out

from pathlib import Path
Lpath=Path(f'{out}/Left.real.npy')
Rpath=Path(f'{out}/Right.real.npy')

Lborder,Rborder=borderROI(subj)

if Lpath.is_file():
    pass 
else:
    Lcanonical=getDist2borderRatio(subj,Lrois[0],Lrois[1],Lborder,'L')
    np.save(f'{out}/Left.real',Lcanonical)
    
    Lgr,Rgr=gradDefROIs(subj)
    LGradDefined=getDist2borderRatio(subj,Lgr[0],Lgr[1],Lborder,'L')
    np.save(f'{out}/Left.GradDefined',LGradDefined)

if Rpath.is_file():
    pass
else:
    Rcanonical=getDist2borderRatio(subj,Rrois[0],Rrois[1],Rborder,'R')
    np.save(f'{out}/Right.real',Rcanonical)
    
    Lgr,Rgr=gradDefROIs(subj)
    RGradDefined=getDist2borderRatio(subj,Rgr[0],Rgr[1],Rborder,'R')
    np.save(f'{out}/Right.GradDefined',RGradDefined)

##### do the permutation measures 
LspinDists=[]
for i in range(len(Lspin)):
    sp=Lspin[i]
    LspinDists.append(getDist2borderRatio(subj,sp[Lrois[0]],sp[Lrois[1]],sp[Lborder],'L'))
LspinDists=np.vstack(LspinDists)

basename_spin=spin.split('/')[1].split('.pickle')[0]
np.save(f'{out}/spinBatches/L.{basename_spin}',LspinDists)

RspinDists=[]
for i in range(len(Rspin)):
    sp=Rspin[i]
    RspinDists.append(getDist2borderRatio(subj,sp[Rrois[0]],sp[Rrois[1]],sp[Rborder],'R'))
RspinDists=np.vstack(RspinDists)

basename_spin=spin.split('/')[1].split('.pickle')[0]
np.save(f'{out}/spinBatches/R.{basename_spin}',RspinDists)
