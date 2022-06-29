#!/usr/bin/env python
import os
import sys
import pathlib
import argparse
import nibabel as nib
import numpy as np
import surfdist
from surfdist import *
import matplotlib.pyplot as plt
import nilearn.plotting 
import hcp_utils as hcp
import subprocess as sp
from dist_utils import * 
print('imports are good')




parser = argparse.ArgumentParser(description='Embeds functional Connectivity Matrix using PCA and diffusion map embedding, and then determines how far peaks are across sessions of the same subject',\
	usage='pkReliability.py --subj <HCP subject> --odir <output directory> ',\
	epilog=("Example usage: "+"pkReliability.py --subj 100206 --odir /well/margulies/projects/pks"),add_help=True)


if len(sys.argv) < 2:
	parser.print_help()
	sys.exit(1)

req_grp = parser.add_argument_group(title='Required arguments')
req_grp.add_argument('--subj',type=str,metavar='',required=True,help='HCP subject to run on')
req_grp.add_argument('--odir',type=str,metavar='',required=True,help='Output directory base. Output will be saved as odir/subj/...')

op_grp= parser.add_argument_group(title='Optional arguments')
args=parser.parse_args()

#get arguments into variables
subj=args.subj
odir=args.odir

### set up subjects output directory 
odir=f'{odir}/{subj}'
print(odir)
os.makedirs(odir,exist_ok=True)

cluster_path='/well/margulies/users/mnk884/data20/'
###### set up files 
subjdir=f'{cluster_path}/{subj}' ### the folder containing the Structural and Rest folders. Change to match cluster when access given
anatdir=f'{subjdir}/Structural'


##### cifti indices 
lcort=slice(0,29696)
rcort=slice(29696, 59412)
cortAll=slice(0,59412)


#### left anatomical files
Lsrf32=f'{anatdir}/{subj}.L.midthickness.32k_fs_LR.surf.gii'
LsrfNative=f'{anatdir}/{subj}.L.midthickness.native.surf.gii'
Lsphere32=f'{anatdir}/{subj}.L.sphere.32k_fs_LR.surf.gii'
LsphereNat=f'{anatdir}/{subj}.L.sphere.reg.reg_LR.native.surf.gii'
Laparc=f'{anatdir}/{subj}.L.aparc.a2009s.native.label.gii'
Laparc32=f'{anatdir}/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'

#### right anatomical files 
Rsrf32=f'{anatdir}/{subj}.R.midthickness.32k_fs_LR.surf.gii'
RsrfNative=f'{anatdir}/{subj}.R.midthickness.native.surf.gii'
Rsphere32=f'{anatdir}/{subj}.R.sphere.32k_fs_LR.surf.gii'
RsphereNat=f'{anatdir}/{subj}.R.sphere.reg.reg_LR.native.surf.gii'
Raparc=f'{anatdir}/{subj}.R.aparc.a2009s.native.label.gii'
Raparc32=f'{anatdir}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'




for hemi in ['L','R']:
    if hemi == 'L':
        ### calc distances
        dists=calc_dists(Laparc,LsrfNative)
        
        ### load meshes to resample dists to 32K 
        s1v=nib.load(Lsrf32).darrays[0].data
        s1f=nib.load(Lsrf32).darrays[1].data
        s2v=nib.load(LsrfNative).darrays[0].data
        s2f=nib.load(LsrfNative).darrays[1].data
        
        nodes=utils.find_node_match(s1v,s2v)
        dist32=dists[nodes[0]]
        np.save(f'/well/margulies/projects/pkReliability/{subj}.{hemi}.dist.native.npy',dists)
        np.save(f'/well/margulies/projects/pkReliability/{subj}.{hemi}.dist32K.npy',dist32)
        
    if hemi =='R':
        ### calc distances
        dists=calc_dists(Raparc,RsrfNative)
        
        ### load meshes to resample dists to 32K 
        s1v=nib.load(Rsrf32).darrays[0].data
        s1f=nib.load(Rsrf32).darrays[1].data
        s2v=nib.load(RsrfNative).darrays[0].data
        s2f=nib.load(RsrfNative).darrays[1].data
        
        nodes=utils.find_node_match(s1v,s2v)
        dist32=dists[nodes[0]]
        
        np.save(f'/well/margulies/projects/pkReliability/{subj}.{hemi}.dist.native.npy',dists)
        np.save(f'/well/margulies/projects/pkReliability/{subj}.{hemi}.dist32K.npy',dist32)
        
