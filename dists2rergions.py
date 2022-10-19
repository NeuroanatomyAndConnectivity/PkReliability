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

def deepS1(subjClass):
    Lsulc=nib.load(subjClass.Lsulc).darrays[0].data
    Ltarg=subjClass.LS1 #[np.argmin(Lsulc[subjClass.LS1])]
    
    Rsulc=nib.load(subjClass.Rsulc).darrays[0].data
    Rtarg=subjClass.RS1 #[np.argmin(Rsulc[subjClass.RS1])]
    return Ltarg,Rtarg
  
inst=hcp_subj(subj,4)
targets=deepS1(inst)
Lsurf=([inst.Lcoords,inst.Lfaces])
Ldist=surfdist.analysis.dist_calc(Lsurf,inst.Lfill,targets[0])
Lfront=Ldist[LfROI]
Lpost=Ldist[LpROI]

Rsurf=([inst.Rcoords,inst.Rfaces])
Rdist=surfdist.analysis.dist_calc(Rsurf,inst.Rfill,targets[1])
Rfront=Rdist[RfROI]
Rpost=Rdist[RpROI]

np.save(f'2DistRois/{subj}.npy',np.asarray([Lfront.mean(),Lpost.mean(),Rfront.mean(),Rpost.mean()]))
