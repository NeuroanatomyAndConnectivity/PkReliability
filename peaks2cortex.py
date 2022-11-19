#! /usr/bin/env python
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
