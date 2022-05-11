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

###### set up files 
subjdir=f'../{subj}/100206' ### the folder containing the Structural and Rest folders. Change to match cluster when access given
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

##### load watershed zone templates ##### place these in a fixed location on the server
LWS=nib.load('../watershedTemplates/LWS.28.max.label.gii').agg_data()
RWS=nib.load('../watershedTemplates/RWS.28.max.label.gii').agg_data()



####### define the functions we'll be using 

def save_gifti(data,out):
	gi = nib.gifti.GiftiImage()
	da = nib.gifti.GiftiDataArray(np.float32(data), intent=0)
	gi.add_gifti_data_array(da)
	nib.save(gi,f'{out}.func.gii')


def post_smooth(func):
	"""Zscore normalize, bandpass filter, and remove first 10 volumes"""
	cifti=nib.load(func)
	#### clean the time series 
	cln=signal.clean(cifti.get_fdata(),detrend=True,standardize='zscore',filter='butterworth',low_pass=0.08,high_pass=0.008)
	return cln[10:]

def wb_smoothCleanTs(func_dat,kernel,leftSrf,rightSrf):
	"""" Smoooth, Normalize and Bandpass Filter data """
	inter=func_dat.split('dtseries.nii')[0]+f'{kernel}mm.dtseries.nii' #### named inter because file will be deleted
	cmd=f'wb_command -cifti-smoothing {func_dat} {kernel} {kernel} COLUMN {inter} -left-surface {leftSrf} -right-surface {rightSrf}'
	sp.run(cmd,shell=True)
	clnTs=post_smooth(inter)
	sp.run(f'rm {inter}',shell=True) 
	return clnTs

def get_corticalVertices(data):
	""" Get indices of Cortex Data from cifti file """
	cifti=nib.load(data)
	structMap=cifti.header.get_index_map(1)
	brainModels=list(structMap.brain_models)
	LCrtBM=brainModels[0]
	Lcrt_vrts=np.array(LCrtBM.vertex_indices)
	LnumVerts=LCrtBM.surface_number_of_vertices
	
	RCrtBM=brainModels[1]
	Rcrt_vrts=np.array(RCrtBM.vertex_indices)
	RnumVerts=RCrtBM.surface_number_of_vertices
	
	return {'lIDX':Lcrt_vrts,'lnverts':LnumVerts,'rIDX':Rcrt_vrts,'rnverts':RnumVerts}

def concat_sessions(DataList):
	return np.vstack([DataList[0],DataList[1]]).T, np.vstack([DataList[2],DataList[3]]).T

def pick_cortex(data,label):
	##### standard slices for HCP data and the 32k surface the time series are mapped on to 
	lcort=slice(0,29696)
	rcort=slice(29696, 59412)
	cortAll=slice(0,59412)
	###### slice time series based on hemisphere choice
	if label=='left':
		print('Using Left Cortex Only')
		data=data[lcort]
	elif label=='right':
		print('Using Right Cortex Only')
		data=data[rcort]
	else:
		print('Using whole cortex')
		data=data[cortAll]
	return data
		
####### use only if running locally. 
def calcFC(data):
	return np.corrcoef(data)


def calcFC_chunks(data):
	bigdata=data
	bigdata -= np.mean(bigdata, axis=1)[:,None]
	bigdata /= np.sqrt(np.sum(bigdata*bigdata, axis=1))[:,None]
	SPLITROWS = 1000
	numrows = bigdata.shape[0]
	res = np.memmap(f'{odir}/tmp.dat', 'float64', mode='w+', shape=(numrows, numrows))
	
	for r in range(0, numrows, SPLITROWS):
		for c in range(0, numrows, SPLITROWS):
			r1 = r + SPLITROWS
			c1 = c + SPLITROWS
			chunk1 = bigdata[r:r1]
			chunk2 = bigdata[c:c1]
			res[r:r1, c:c1] = np.dot(chunk1, chunk2.T)
	return res

def threshMat(conn,lim):
	perc = np.array([np.percentile(x, lim) for x in conn])
	# Threshold each row of the matrix by setting values below X percentile to 0
	for i in range(conn.shape[0]):
		conn[i, conn[i,:] < perc[i]] = 0   
	return conn

def pcaGrad(data):
	pca = GradientMaps(n_components=1, random_state=0,approach='pca')
	pca.fit(data)
	return pca.gradients_[:].squeeze()


def DiffEmbed(data):
	####input is threshold FC matrix
	aff = 1 - pairwise_distances(data, metric = 'cosine')
	dm = GradientMaps(n_components=1, random_state=0,approach='dm')
	dm.fit(aff)
	return dm.gradients_[:].squeeze()



def save_grads(pcaMap,deMap,idxMap,session,hemi=''):
    if hemi=='left':
        lpc=np.zeros(idxMap[0]['lnverts'])
        lpc[idxMap[0]['lIDX']]=pcaMap
        save_gifti(lpc,f'{odir}/{subj}.L.PCA.G1.{kernel}mmTsSes{session}')
        
        lde=np.zeros(idxMap[0]['lnverts'])
        lde[idxMap[0]['lIDX']]=deMap
        save_gifti(lde,f'{odir}/{subj}.L.DE.G1.{kernel}mmTsSes{session}')
        return lpc,lde
    elif hemi=='right':
        rpc=np.zeros(idxMap[0]['rnverts'])
        rpc[idxMap[0]['rIDX']]=pcaMap
        save_gifti(rpc,f'{odir}/{subj}.R.PCA.G1.{kernel}mmTsSes{session}')
        
        rde=np.zeros(idxMap[0]['rnverts'])
        rde[idxMap[0]['rIDX']]=deMap
        save_gifti(rde,f'{odir}/{subj}.R.DE.G1.{kernel}mmTsSes{session}')
        return rpc,rde
    else: 
        lpcaMap=pcaMap[lcort]
        ldeMap=deMap[lcort]
        
        lpc=np.zeros(idxMap[0]['lnverts'])
        lpc[idxMap[0]['lIDX']]=lpcaMap
        save_gifti(lpc,f'{odir}/{subj}.L.PCA.G1.{kernel}mmTsSes{session}')
        
        lde=np.zeros(idxMap[0]['lnverts'])
        lde[idxMap[0]['lIDX']]=ldeMap
        save_gifti(lde,f'{odir}/{subj}.L.DE.G1.{kernel}mmTsSes{session}')
        
        rpcaMap=pcaMap[rcort]
        rdeMap=deMap[rcort]
        rpc=np.zeros(idxMap[0]['rnverts'])
        rpc[idxMap[0]['rIDX']]=rpcaMap
        save_gifti(rpc,f'{odir}/{subj}.R.PCA.G1.{kernel}mmTsSes{session}')
        
        rde=np.zeros(idxMap[0]['rnverts'])
        rde[idxMap[0]['rIDX']]=rdeMap
        
        save_gifti(rde,f'{odir}/{subj}.R.DE.G1.{kernel}mmTsSes{session}')
        return lpc,lde,rpc,rde


def gradientOrientation(grad,hemi):
	grad=nib.load(grad).agg_data()
	if hemi=='left':
		print('running left')
		labels=nib.load(Laparc).agg_data()
		calc=np.where(labels==45)[0]
		ctr=np.where(labels==46)[0]
		if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
			print('Canonical Orientation DMN at apex')
			return grad
		elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
			print(f'{subj} has flipped gradient ordering. remove from study')
		else:
			print('flipping gradient orientation for peak detection')
			return grad *-1
		
	elif hemi=='right':
		print('running right')
		labels=nib.load(Raparc)
		calc=np.where(labels==45)[0]
		ctr=np.where(labels==46)[0]
		if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
			print('Canonical Orientation DMN at apex')
			return grad
		elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
			print(f'{subj} has flipped gradient ordering. remove from study')
			return grad
		else:
			print('flipping gradient orientation before peak detection')
			return grad *-1


def get_peaks(grad,zoneParc):
	labels=zoneParc
	Lpar=np.where(labels==2)[0]
	Ltmp=np.where(labels==5)[0]
	Mpar=np.where(labels==7)[0]
	pks=[]
	for i in [Lpar,Ltmp,Mpar]:
		pks.append(i[np.argmax(grad[i])])
	return pks

def find_nat_pk(pks,surf1,surf2):
	surf1=nib.load(surf1).darrays[0].data
	surf2=nib.load(surf2).darrays[0].data

	srf_mathced=find_node_match(surf1,surf2)
	
	natVrt=[]
	for i in pks:
		natVrt.append(srf_mathced[i])
	return np.asarray(natVrt)

def dist_btw_pks(set1,set2,surf):
	verts=nib.load(surf).darrays[0].data.astype('float64')
	faces=nib.load(surf).darrays[1].data.astype('int32')
	for i,j in zip(set1,set2):
		i=np.asarray([i]).astype('int32')
		j=np.asarray([j]).astype('int32')
		print(gd.compute_gdist(verts,faces,i,j))


######## do the thing 
func_ses411=[]
for data in range(len(func_ses)):
	#### start by getting the indices of cortical vertices
	func_ses411.append(get_corticalVertices(func_ses[data]))
	##### smooth and clean the funcitonal time series

if pks==False:
	kernel=5.0 #### comment in full list when done 
	func_ses411=[]
	for data in range(len(func_ses)):
		##### smooth and clean the funcitonal time series
		func_ses[data]=wb_smoothCleanTs(func_ses[data],kernel,Lsrf32,Rsrf32)
	#### concat the time series
	ses1,ses2=concat_sessions(func_ses)

	### clear memory
	del func_ses

	sessions=[ses1,ses2]
	######### build the functional connectivity matrix ######### 
	x=0
	for ses in sessions:
		x=x+1

		if local==True:
			dconn=calcFC_chunks(pick_cortex(ses,hemi))
			print(dconn.shape)
		else:
			dconn=calcFC(pick_cortex(ses,hemi))
			print(dconn.shape)

		#### threshold the connectivity matrix 
		dconn=threshMat(dconn,95)

		##### do PCA on the thresholded connectivity matrix 
		print('running pca')
		pcaG1=pcaGrad(dconn)

		#### do diffusion embedding on the cosine Affinity matrix of dconn
		print('running diffusion maps')
		dmG1=DiffEmbed(dconn)


		pc32,de32=save_grads(pcaG1,dmG1,func_ses411,f'0{x}','left')

		### clear memory 
		del dconn 

	#### now we do the peak distances across sessions
	#### we exit the for loop and the code here is now the same as running the --pks option


else:
	print('we have grads already. hell yeah ')






