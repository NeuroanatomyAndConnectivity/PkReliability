from hcp_class import *
import pickle
import surfdist
import subprocess as sp
import surfdist.analysis

##### function to save giftis for probablility maps 
def save_gifti(data,out):
	"""Save gifti file providing a numpy array and an output file path"""
	gi = nib.gifti.GiftiImage()
	da = nib.gifti.GiftiDataArray(np.float32(data), intent=0)
	gi.add_gifti_data_array(da)
	nib.save(gi,f'{out}.func.gii')


with open('SubjectsCompleteData.txt') as f:
    subjects=f.read()
subjects=subjects.split('\n')
subjects.pop()

sub_dict={}
for i in subjects:
    sub_dict[i]=hcp_subj(i,4)
    
for key in sub_dict.copy():
    if sub_dict[key].Rgrad[1] != True or sub_dict[key].Lgrad[1] !=True:
#         print(key)
        del sub_dict[key]
  
  
  
  with open('subDict04mm.pickle', 'wb') as handle:
    pickle.dump(sub_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  
  
  threshold=90
  
  
 #### get gradient probability map 
probmapL=[]
probmapR=[]
for key in sub_dict.copy():
    grads=sub_dict[key].extract_topX(sub_dict[key].Lgrad,sub_dict[key].Rgrad,threshold)
    lmap=np.zeros(sub_dict[key].dims)
    lmap[grads[0]]=1
    probmapL.append(lmap)
    rmap=np.zeros(sub_dict[key].dims)
    rmap[grads[1]]=1
    probmapR.append(rmap)
probmapL=np.vstack(probmapL).sum(axis=0)/len(probmapL)
probmapR=np.vstack(probmapR).sum(axis=0)/len(probmapR)

save_gifti(probmapL,'L.GradProbMap')
save_gifti(probmapR,'R.GradProbMap')



