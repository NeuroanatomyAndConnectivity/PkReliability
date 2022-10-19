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
    print(f'left target is S1 with shape {Ltarg.shape}')
    Rsulc=nib.load(subjClass.Rsulc).darrays[0].data
    Rtarg=subjClass.RS1 #[np.argmin(Rsulc[subjClass.RS1])]
    print(f'right target is S1 with shape {Rtarg.shape}')
    return Ltarg,Rtarg
  
inst=hcp_subj(subj,4)
targets=deepS1(inst)
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
