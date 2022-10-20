from hcp_class import hcp_subj
import surfdist.analysis
import nibabel as nib
from nilearn.plotting import view_surf
import numpy as np
import sys

subj=sys.argv[1]
print(f'calculating distance from gradient mask to  parcels for subject {subj}')

### load the defined ROIs
Lfroi=np.load('2DistRois/LfRoi.npy')
Lproi=np.load('2DistRois/LpRoi.npy')
Rfroi=np.load('2DistRois/RfRoi.npy')
Rproi=np.load('2DistRois/RpRoi.npy')

#### get's the nearest neighbours of each vertex in a label
def get_labelNN(coords,faces,vertices):
    mini_dict={}
    for i in vertices:
        a=np.unique(faces[np.where(faces==i)[0]])
        a=a[a!=i]
        mini_dict[i]=a
    return mini_dict 
#### takes the vertices associated with two labels and returns the shared border between the two 
def borders_betweenLabels(subj,hemi,label1verts,label2verts):
    if hemi =='L':
        coords=subj.Lcoords
        faces=subj.Lfaces
        l1=get_labelNN(coords,faces,label1verts)
        l1=np.unique(list(l1.values()))
        l2=get_labelNN(coords,faces,label2verts)
        l2=np.unique(list(l2.values()))
        return np.intersect1d(l1,l2)
    elif hemi =='R':
        coords=subj.Rcoords
        faces=subj.Rfaces
        l1=get_labelNN(coords,faces,label1verts)
        l1=np.unique(list(l1.values()))
        l2=get_labelNN(coords,faces,label2verts)
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
  
inst=hcp_subj(subj,4)
targets=borderROI(inst)
Lsurf=([inst.Lcoords,inst.Lfaces])
Ldist=surfdist.analysis.dist_calc(Lsurf,inst.Lfill,targets[0])
Lfront=Ldist[Lfroi]
Lpost=Ldist[Lproi]
Ldist1=surfdist.analysis.dist_calc(Lsurf,inst.Lfill,Lfroi)
Ldist2=surfdist.analysis.dist_calc(Lsurf,inst.Lfill,Lproi)
LminPkdist=np.min(np.vstack([Ldist1,Ldist2]),axis=0)

Rsurf=([inst.Rcoords,inst.Rfaces])
Rdist=surfdist.analysis.dist_calc(Rsurf,inst.Rfill,targets[1])
Rfront=Rdist[Rfroi]
Rpost=Rdist[Rproi]

Rdist1=surfdist.analysis.dist_calc(Rsurf,inst.Rfill,Rfroi)
Rdist2=surfdist.analysis.dist_calc(Rsurf,inst.Rfill,Rproi)
RminPkdist=np.min(np.vstack([Rdist1,Rdist2]),axis=0)

np.save(f'2DistRois/{subj}.npy',np.vstack([Ldist,LminPkdist,Rdist,RminPkdist]))
