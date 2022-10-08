from hcp_class import *
import pickle
import surfdist
import subprocess as sp
import surfdist.analysis
import sys

subj=sys.argv[1]
print(f'calculating distance from gradient mask to  parcels for subject {subj}')

def DistFromGradMask(subj,threshold):
	subj_inst=hcp_subj(subj,4)
	grads=subj_inst.extract_topX(subj_inst.Lgrad,subj_inst.Rgrad,threshold)
	Lsurf=[subj_inst.Lcoords,subj_inst.Lfaces]
	Ldist=surfdist.analysis.dist_calc(Lsurf,subj_inst.Lfill,grads[0])
	Rsurf=[subj_inst.Rcoords,subj_inst.Rfaces]
	Rdist=surfdist.analysis.dist_calc(Rsurf,subj_inst.Rfill,grads[1])
	##### get the surface areas -- sqrt(sum of cortical vertex areas) 
	sp.run(f'wb_command -surface-vertex-areas {subj_inst.Lsrf} tmp/{subj}.L.area.func.gii',shell=True)
	Larea=np.sqrt(np.sum(nib.load(f'tmp/{subj}.L.area.func.gii').darrays[0].data[subj_inst.Lfill]))
	sp.run('rm L.area.func.gii',shell=True)
	sp.run(f'wb_command -surface-vertex-areas {subj_inst.Rsrf} tmp/{subj}.R.area.func.gii',shell=True)
	Rarea=np.sqrt(np.sum(nib.load(f'tmp/{subj}.R.area.func.gii').darrays[0].data[subj_inst.Rfill]))
	sp.run('rm R.area.func.gii',shell=True)
	return [Ldist,Larea,Rdist,Rarea]

results=DistFromGradMask(subj,90)

with open(f"results/distancesFromGradMask/{subj}.DistFromGrad.pickle", "wb") as dist:
	pickle.dump(results, dist)