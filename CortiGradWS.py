#!/usr/bin/env python

import os
import sys
import subprocess as sp
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import numba
from statistics import mode
from mapalign import embed
from sklearn.metrics import pairwise_distances
from statistics import mode


### let's parse our arguments and create options 
### waterhsed needs 
#### W a metric file to parcellate - either cifti or gifti 
######### Medial wall should be set to zero prior to running. don't run with medial wall == NAN
### a coritcal surface file 
### a method defined as 'min' or 'max'
### a radius -- r will only be used on the actual geodesic distance bit 
#### option -- conn where a time series or blueprint is input, a distance matrix is calculated
##              vertices are selected by whether or not they suvive said threshold



### get arguments and parse them
parser = argparse.ArgumentParser(description='Generate png slices of anatomical image at center of gravity',usage='quick_look.py -a < my anat> -m <my mask> ',epilog=("Example usage: "+"quick_look.py -a anat.nii.gz "),add_help=False)
if len(sys.argv) < 1:
    parser.print_help()
    sys.exit(1)

req_grp = parser.add_argument_group(title='Required arguments')
req_grp.add_argument('-s','--subj',type=str,metavar='',required=True,help='left functional time series')
req_grp.add_argument('-t','--thr',type=str,metavar='',required=True,help='Connectivitythreshold for embedding')
req_grp.add_argument('-r','--radius',type=str,metavar='',required=True,help=' radius to search the geodesic distance in the Watershed')
req_grp.add_argument('-m','--method',type=str,metavar='',required=True,help=' specifiy min or max to use in watershed')
# req_grp.add_argument('-r','--rfunc',type=str,metavar='',required=True,help='right funcitonal data')

op_grp= parser.add_argument_group(title='Optional arguments')
# op_grp.add_argument('-l','--alt',type=str,metavar='',help='Resample data to alternate reference image for slice represenation. Generally used in animal studies,args.alt')
# op_grp.add_argument('-t','--thr',type=float,metavar='',help='Minimum threshold applied to statistical image. between 0-1. Default is 0.2')
# #op_grp.add_argument('-o','--out',type=str,metavar='',help='Specify output directory. Defualt is cwd/figures')
op_grp.add_argument("-h", "--help", action="help", help="show this help message and exit")
args=parser.parse_args()





#get those arguments into variables
subj=args.subj
print(f'running on {subj}')
thresh=args.thr
radius=args.radius
method=args.method
# #### load them niftis in
# anat_obj=nib.load(anat,mmap=False)
# anat_dat = anat_obj.get_data().astype(float)
# anat_geom=anat_dat.shape
# #### load mask or statistical map into 
# mask_obj=nib.load(mask,mmap=False)
# mask_dat= mask_obj.get_data().astype(float)
# mask_geom=mask_dat.shape
# mask_name=os.path.basename(mask)
# print(mask_name)



##### define fucntions used in pipeline 

### self explanatory no?
def save_gifti(data,out):
    gi = nib.gifti.GiftiImage()
    da = nib.gifti.GiftiDataArray(np.float32(data), intent=0)
    gi.add_gifti_data_array(da)
    nib.save(gi,f'{out}.func.gii')

##### threshold connectome to make affinity matrix for embedding
def threshold_connectome(A,per):
#     if type(A) != 'numpy.ndarray':
#         np.fill_diagonal(A.values, 1)
#         A=A.values
        
    A_cp=A.copy()
    per=int(per)
    perc = np.array([np.percentile(x, per) for x in A_cp])
    
    for i in range(A_cp.shape[0]):
        A_cp[i, A_cp[i,:] < perc[i]] = 0
    A_cp[A_cp<0]=0
    return A_cp

##### place subcortex back into output vector.
def placeSC(vec,SC):
    ##### places the subcortex back into a vector before saving
    vec=list(vec)
    for i in SC:
        vec.insert(i,0)
    return vec

###### take a left fmri time series and right, combine, them, mask out subC and get corr matrix
def indiConnMat(L,R):
    ### load the giftis
    Left=nib.load(L)
    Right=nib.load(R)
    
    ### get the data into a np array
    ldat=data=[ Left.darrays[i].data for i in range(len(Left.darrays))]
    ldat=np.vstack(ldat).T
    lsC=np.where(np.sum(ldat,axis=1)==0)[0]
    lcort=np.where(np.sum(ldat,axis=1)!=0)[0]
    
    
    rdat=data=[ Right.darrays[i].data for i in range(len(Right.darrays))]
    rdat=np.vstack(rdat).T
    rsC=np.where(np.sum(rdat,axis=1)==0)[0]
    rcort=np.where(np.sum(rdat,axis=1)!=0)[0]
    
    #### full cortex arrays
    fullC=np.vstack([ldat,rdat])
    offset=int(len(fullC)/2)
    
    ### get subcortical and cortical vertices
    cort=np.where(np.sum(fullC,axis=1)!=0)[0]
    subC=np.where(np.sum(fullC,axis=1)==0)[0]
    
    
    cortexLR=fullC[cort]
    
    conn=np.corrcoef(cortexLR)
    
    return conn,cort,subC,offset,lsC,rsC,lcort,rcort


##### this is the watershed function we call 
def cortiGradWS(W,D,m,method):
    ##### adapted from eyal soreq's matlab code
    #### W is masked metric file to use in watershed
    #### D is masked distance matrix for a previously specified radius
    m=m
    #### get the order of the elements of local minima or the maximum depending on method provided
    idx=np.argsort(W)
    if method =='min':
        idx=idx
    elif method =='max':
        idx=idx[::-1]
    
    #### need to go in order from most to least not on the distL alone
    #### so reorder the distL to go by idx
    
    
    N= len(idx)
    C=np.zeros(W.shape)
    for ii in idx:
        ##### find where the nodes in your ROI are
        subNodes=np.where(D[ii]>0)[0]
        c=C[subNodes]
        c=c[c>0]
        
        ##### next part conditional
        
        if c.size == 0:
            C[ii]=m
            m=m+1
            ## comment back in. this is to see if you can plot just the watershed itself
        elif ~np.any(np.diff(c)):
            C[ii]=c[0]
        else:
            C[ii]=mode(c)
    return C


### path to functional file parent directory
fpath=f'Rest/{subj}/MNINonLinear/Results/rfMRI_REST1_LR/'


print('Creating Connectivity Matrix')
#### r = corr mat, ### ctx = cortex, subc = subcortex, lsC/rsC = left and right isolates subcortex, l/rcort = left and right isolated cortival vertices
##### this is code very specific to HCP and this test run. 
##### ideally create paths to each file as required option so it can be run more broadly
r,ctx,subc,offset,lsC,rsC,lcort,rcort=indiConnMat(f'{fpath}rfMRI_REST1.L.10K.func.gii',f'{fpath}rfMRI_REST1.R.10K.func.gii')
# print('Connectivity Matrix Created.')


# print('embedding with mapalign')

rThr=threshold_connectome(r,thresh)
aff = 1 - pairwise_distances(rThr, metric = 'cosine')
emb, res = embed.compute_diffusion_map(aff, alpha = 0.5,n_components=5,return_result=True,skip_checks=True)

print('embedding completed')


print('saving first two gradient giftis')


LG1=f'{fpath}rfMRI_REST1.L.10KGrad01'
RG1=f'{fpath}rfMRI_REST1.R.10KGrad01'

LG2=f'{fpath}rfMRI_REST1.L.10KGrad02'
RG2=f'{fpath}rfMRI_REST1.L.10KGrad02'


save_gifti(placeSC(emb[:,0],subc)[0:offset],LG1)
save_gifti(placeSC(emb[:,0],subc)[offset:],RG1)


save_gifti(placeSC(emb[:,1],subc)[0:offset],LG2)
save_gifti(placeSC(emb[:,1],subc)[offset:],RG2)


##### smooth the files we just created
#### get our anatomical surfaces paths loaded

### path to structural file parent directory 
spath=f'Structural/{subj}/MNINonLinear/fsaverage_LR32k/'

structL=f'{spath}{subj}.L.midthickness_MSMAll.10k_fs_LR.surf.gii'
structR=f'{spath}{subj}.R.midthickness_MSMAll.10k_fs_LR.surf.gii'


print('run the watershed algorithm on the individual gradient')
Lgrads=[LG1,LG2]
for grad in Lgrads:
    print(f'smoothing {grad} by 6mm smoothing kernel')
    spobj=sp.run(f'wb_command -metric-smoothing {structL} {grad}.func.gii 6 {grad}.func.gii',shell=True)
    
    W=nib.load(f'{grad}.func.gii').darrays[0].data
    W[lsC]=0
    WL=W[lcort]

    print('get the geodesic all-to-all distance matrix')
    os.getcwd()
    spobj=sp.run(f'wb_command -surface-geodesic-distance-all-to-all {structL} -limit {radius} dist{radius}.dconn.nii',shell=True)
    dist=nib.load(f'dist{radius}.dconn.nii').get_fdata()
    os.remove(f'./dist{radius}.dconn.nii')
    dist=dist[lcort,:][:,lcort]

    print('doing the watershed')
    LabelVector=cortiGradWS(WL,dist,1,method)


    out=placeSC(LabelVector,lsC)

    save_gifti(out,f'{grad}.{method}.{radius}')

    sp.run(f'wb_command -set-structure {grad}.{method}.{radius}.func.gii CORTEX_LEFT',shell=True)


print('Doing the right thing.')
Rgrads=[RG1,RG2]
for grad in Rgrads:
    print(f'smoothing {grad} by 6mm smoothing kernel')
    spobj=sp.run(f'wb_command -metric-smoothing {structR} {grad}.func.gii 6 {grad}.func.gii',shell=True)
    
    W=nib.load(f'{grad}.func.gii').darrays[0].data
    W[rsC]=0
    WL=W[rcort]

    print('get the geodesic all-to-all distance matrix')
    os.getcwd()
    spobj=sp.run(f'wb_command -surface-geodesic-distance-all-to-all {structR} -limit {radius} dist{radius}.dconn.nii',shell=True)
    dist=nib.load(f'dist{radius}.dconn.nii').get_fdata()
    os.remove(f'./dist{radius}.dconn.nii')
    dist=dist[rcort,:][:,rcort]

    print('doing the watershed')
    LabelVector=cortiGradWS(WL,dist,1,method)


    out=placeSC(LabelVector,rsC)

    save_gifti(out,f'{grad}.{method}.{radius}')

    sp.run(f'wb_command -set-structure {grad}.{method}.{radius}.func.gii CORTEX_RIGHT',shell=True)



