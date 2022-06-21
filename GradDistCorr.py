#!/usr/bin/env python
import os
import sys
import pathlib
import argparse
import nibabel as nib
import numpy as np
import brainspace as bs
import subprocess as sp 
import cython
from nilearn import signal
from scipy import spatial
import gdist as gd
from brainspace.gradient import GradientMaps
from sklearn.metrics import pairwise_distances
from surfdist.utils import find_node_match
from utils import *

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
op_grp.add_argument('--pks',action='store_true',help='Only do peak detection. Assumes embedding has been run previously')
op_grp.add_argument('--local',action='store_true',help='Chunked calculation of Functional Connectivity Matrix using local HD or ramdisk to manage memory')
op_grp.add_argument('--hemi',choices=['left','right'],type=str,metavar='',help='Default All. Otherwise specify \"left" or\"right".')
# op_grp.add_argument("-h", "--help", action="help", help="show this help message and exit")
args=parser.parse_args()

#get arguments into variables
subj=args.subj
odir=args.odir
pks=args.pks
local=args.local
hemi=args.hemi
if hemi == None:
	hemi='all'
print(f'do peaks only is {pks}')
print(f'running on {hemi} hemisphere(s)')

### set up subjects output directory 
odir=f'{odir}/{subj}'
print(odir)
os.makedirs(odir,exist_ok=True)

cluster_path='/well/margulies/users/mnk884'
###### set up files 
subjdir=f'{cluster_path}/{subj}/{subj}' ### the folder containing the Structural and Rest folders. Change to match cluster when access given
fdir=f'{subjdir}/Rest'
anatdir=f'{subjdir}/Structural'


##### resting state time series paths 
ses1LR=f'{fdir}/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii'
ses2LR=f'{fdir}/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii'
ses1RL=f'{fdir}/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii'
ses2RL=f'{fdir}/rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'
func_ses=[ses1LR,ses2RL,ses1RL,ses2LR]


#### left anatomical files
Lsrf32=f'{anatdir}/{subj}.L.midthickness.32k_fs_LR.surf.gii'
LsrfNative=f'{anatdir}/{subj}.L.midthickness.native.surf.gii'
Lsphere32=f'{anatdir}/{subj}.L.sphere.32k_fs_LR.surf.gii'
LsphereNat=f'{anatdir}/{subj}.L.sphere.reg.reg_LR.native.surf.gii'
Laparc=f'{anatdir}/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'

#### right anatomical files 
Rsrf32=f'{anatdir}/{subj}.R.midthickness.32k_fs_LR.surf.gii'
RsrfNative=f'{anatdir}/{subj}.R.midthickness.native.surf.gii'
Rsphere32=f'{anatdir}/{subj}.R.sphere.32k_fs_LR.surf.gii'
RsphereNat=f'{anatdir}/{subj}.R.sphere.reg.reg_LR.native.surf.gii'
Raparc=f'{anatdir}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'


print(Lsrf32,Rsrf32)