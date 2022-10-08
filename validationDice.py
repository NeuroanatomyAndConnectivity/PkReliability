import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle

def recort(X,fill,dims):
    out=np.zeros(dims)
    out[fill]=X
    return out

def gradientOrientation(grad,hemi,aparc):
    """Determine the orientation of the gradients, and also return whether valid for continued study or not"""
    grad=grad #nib.load(grad).agg_data()
    if hemi=='left':
        labels=nib.load(aparc).agg_data()
#         print('getting gradient orientation from left hemisphere')
    else:
        labels=nib.load(aparc).agg_data()
#         print('getting gradient orientation from right hemisphere')
    calc=np.where(labels==45)[0]
    ctr=np.where(labels==46)[0]
    if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
#         print('Canonical Orientation DMN at apex')
        return grad,True
    elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
#         print(f'REMOVE {subj} FROM STUDY')
        return grad,False
    elif np.sum(grad[calc])>0 and np.sum(grad[ctr])<0:
#         print(f'REMOVE {subj} FROM STUDY')
        return grad,False
    else:
#         print('flipping gradient orientation for peak detection')
        return grad *-1,True

def dice_it(A,B):
    
    num=2*(len(np.intersect1d(A,B)))
    den=len(A)+len(B)
    
    if den ==0:
        return np.nan
    else:
        return num/den


class hcp_subj:

    def __init__(self,subj,kernel,pca=None,neighbours=None):
        
        self.subj=subj
        
        clusterPath='/well/margulies/projects/data/hcpGrads'
        anat32Path=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/fsaverage_LR32k'
        MNIpath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/MNINonLinear/fsaverage_LR32k'
        
        self.info=np.load(f'{clusterPath}/{subj}/{subj}.cifti.info.npy',allow_pickle=True).item()
        
        self.dims=self.info['lnverts']
        self.Lfill=self.info['lIDX']
        self.Rfill=self.info['rIDX']
        self.pca=pca
        self.neighbours=neighbours
        
        self.Lsrf=f'{anat32Path}/{subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.Lcoords=nib.load(self.Lsrf).darrays[0].data
        self.Lfaces=nib.load(self.Lsrf).darrays[1].data
        
        self.Linflated=f'{anat32Path}/{subj}.L.inflated_MSMAll.32k_fs_LR.surf.gii'
        
        
        self.Rsrf=f'{anat32Path}/{subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.Rcoords=nib.load(self.Rsrf).darrays[0].data
        self.Rfaces=nib.load(self.Rsrf).darrays[1].data

        

        self.Laparc=f'{MNIpath}/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'
        
        self.LV1=np.where(nib.load(self.Laparc).darrays[0].data==45)[0]
        self.LS1=np.where(nib.load(self.Laparc).darrays[0].data==46)[0]
        self.LA1=np.where(nib.load(self.Laparc).darrays[0].data==75)[0]
        
        
        self.Rinflated=f'{anat32Path}/{subj}.R.inflated_MSMAll.32k_fs_LR.surf.gii'

        
        self.Raparc=f'{MNIpath}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'
        
        self.RV1=np.where(nib.load(self.Raparc).darrays[0].data==45)[0]
        self.RS1=np.where(nib.load(self.Raparc).darrays[0].data==46)[0]
        self.RA1=np.where(nib.load(self.Raparc).darrays[0].data==75)[0]
        
        
    
#         self.LZverts=get_zoneVerts(LWS)
#         self.RZverts=get_zoneVerts(RWS)
    
#         self.LdistSens=np.load(f'{subj}/{subj}.L.dist32K.npy')
#         self.RdistSens=np.load(f'{subj}/{subj}.R.dist32K.npy')
        
        neighbours=self.neighbours
        
        if self.neighbours==None:
            pass
        else:
            self.Lneighbours=SpatialNeighbours(self.Lcoords,self.Lfaces)
            self.Rneighbours=SpatialNeighbours(self.Rcoords,self.Rfaces)
        
        
        if self.pca is None:
           #print('ussing diffusion maps')

            #### full gradient 
            self.grad=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.diffmaps.0{kernel}mm.npy')
            self.Lgrad=self.grad[0][0:len(self.Lfill)]
            self.Lgrad=recort(self.Lgrad,self.Lfill,self.dims)
            self.Lgrad=gradientOrientation(self.Lgrad,'left',self.Laparc)


            self.Rgrad=self.grad[0][len(self.Lfill):]
            self.Rgrad=recort(self.Rgrad,self.Rfill,self.dims)
            self.Rgrad=gradientOrientation(self.Rgrad,'right',self.Raparc)

            ###### session 1 
            ### subsessions
            self.gradses1=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            ######## session 2 
 
        
            self.gradses2=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')
        
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
            
        else:
#             print('using PCA maps')
            ######### load PCA grads
            self.gradses1=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses1.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            self.gradses2=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses2.s0{kernel}mm.npy')   
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
        
        
        
    
    def print_subj(self):
        print(self.subj)
    
    
    def extract_topX(self,Left,Right,pct):
        """extract the top X percent instead of binning"""
        
        
        Left=Left[0]
        Right=Right[0]
        Lout=np.zeros(self.dims)
        Rout=np.zeros(self.dims)
        
        Lpct=np.percentile(Left[self.Lfill],pct)
        
        
        Lthr=np.where(Left[self.Lfill]>Lpct)[0]
        Linter=np.zeros(len(self.Lfill))
        Linter[Lthr]=1
        L=recort(Linter,self.Lfill,self.dims)
        L=np.where(L!=0)[0]
        
        #### do right 
        
                
        Rpct=np.percentile(Right[self.Rfill],pct)
        
        
        Rthr=np.where(Right[self.Rfill]>Rpct)[0]
        Rinter=np.zeros(len(self.Rfill))
        Rinter[Rthr]=1
        R=recort(Rinter,self.Rfill,self.dims)
        R=np.where(R!=0)[0]
        

        return L,R
 
  
    
    def dice_Ses12(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
      
        diceL=dice_it(S1[0],S2[0])
        diceR=dice_it(S1[1],S2[1])
        
        
        return np.asarray([diceL,diceR])
    
    

with open('SubjectsCompleteData.txt') as file:
    subjects=file.read()
subjects=subjects.split('\n')
subjects.pop()

print(f'using {len(subjects)} of subjects')

def prep_plotting(subj,kernel,sim='dice',pca=False):
#     print(subj)
    
    thr=[50,55,60,65,70,75,80,85,90,95]
    
    ctx_metric=[]
    
    if pca == False:
        gr=hcp_subj(subj,kernel)
        if gr.Lgradses1[1] == False or gr.Lgradses2[1] == False or gr.Rgradses1[1] ==False or gr.Rgradses2[1] ==False:
#             print(f'subject {gr.subj} Diffusion Mapping is not valid at smoothing kernel {kernel} ')
            return [gr.subj,kernel],[gr.subj,kernel]
        else:
            for t in thr:
                ctx_metric.append(gr.dice_Ses12(t))
        
    else:
        gr=hcp_subj(subj,kernel,pca=True)
        if gr.Lgradses1[1] == False or gr.Lgradses2[1] == False or gr.Rgradses1[1] ==False or gr.Rgradses2[1] ==False:
#             print(f'subject {gr.subj} PCA is not valid at smoothing kernel {kernel} ')
            return [gr.subj,kernel],[gr.subj,kernel]
        else:
            for t in thr:
                ctx_metric.append(gr.dice_Ses12(t))
                    
    return np.vstack(ctx_metric)

def prep_plotsXkernel(subj_list,kernel,pca=False,corr=True):
    ### set up outputs
    Lhemi=[]
    Rhemi=[]
    nogo=[]
    
    for subj in subj_list:
        a=prep_plotting(subj,kernel,pca=pca)
        if len(a)>2:
            Lhemi.append(a[:,0])
            Rhemi.append(a[:,1])
        else:
            nogo.append(a[0])
    Lhemi=np.vstack(Lhemi)
    Rhemi=np.vstack(Rhemi)
    if corr == True:
        corr=(len(Lhemi)/len(subj_list))
        return   (Lhemi*corr),(Rhemi*corr),nogo
    elif corr == False:
        return  Lhemi,Rhemi,nogo


#### get rejects
pc_reject=[]
dm_reject=[]

dm_dirtySubj=[]
pc_dirtySubj=[]
kernels=[2,4,6,8,10]
for k in kernels:
    a=prep_plotsXkernel(subjects,k)
    dm_reject.append(len(subjects)-len(a[2]))
    dm_dirtySubj.append(a[-1])
    
    b=prep_plotsXkernel(subjects,k,pca=True)
    pc_reject.append(len(subjects)-len(b[2]))
    pc_dirtySubj.append(b[2])
    
dm_reject=np.asarray(dm_reject)
pc_reject=np.asarray(pc_reject)


with open("pc_rejects", "wb") as pc_rej:   #Pickling
    pickle.dump(pc_dirtySubj, pc_rej)
    
with open("dm_rejects", "wb") as dm_rej:   #Pickling
    pickle.dump(dm_dirtySubj, dm_rej)


# # sn.set()
g=sn.lineplot(x=kernels,y=dm_reject,markers=True, dashes=True,marker='o',label='Dmap')
g=sn.lineplot(x=kernels,y=pc_reject,markers=True, dashes=True,marker='o',label='PCA')
g.set_xticks(kernels)
g.set_xlabel('Smoothing kernel')
g.set_ylabel('Subjects included in Dice')
g.set_title('Canonical first gradient')
plt.savefig('subjects_perkernel.png',facecolor='w')


### set up color palletes and thresholds
thr=[50,55,60,65,70,75,80,85,90,95]
pal = sn.color_palette(palette='Set2',n_colors=10)
### set up the dictionaries so we can plot them with hue's in seaborn 
regions={}
regions['Left Hemisphere']=0
regions['Right Hemisphere']=1
kernels=[2, 4, 6, 8, 10]

def plot_itHue(subjlist,k,rgn,legend=False,corr=True):
    """where k = kernel of 2,4,6,8,10 and rgn indexes the output of prep"""
    if corr==True:
        a=prep_plotsXkernel(subjlist,k)
        b=prep_plotsXkernel(subjlist,k,pca=True)
    elif corr==False:
        a=prep_plotsXkernel(subjlist,k,corr=False)
        b=prep_plotsXkernel(subjlist,k,pca=True,corr=False)

    a=pd.DataFrame.from_dict(dict(zip(thr,a[rgn].T)))
    a['Method']='Diffusion Mapping'
    b=pd.DataFrame.from_dict(dict(zip(thr,b[rgn].T)))
    b['Method']='PCA'

#     print(f'Dmaps has {len(a)} subjects')
#     print(f'PCA has {len(b)} subjects')
#     print(f'smoothing kernel is {k}')
    df=pd.concat([a,b])
    df=df.melt(id_vars=['Method'],value_vars=[50,55,60,65,70,75,80,85,90,95])
#     f, ax = plt.figure()
    ax=sn.boxplot(data=df,x='value',y='variable',hue='Method',orient='h')
    ax=sn.stripplot(data=df,x='value',y='variable',hue='Method',orient='h',size=3,dodge=True,palette=pal)
    
    plt.xlim([0,1])
    ax.set(ylabel = "Gradient Threshold")
    plt.xlabel(f"{k}mm Smoothing.\n Dmap with {len(a)}/{len(subjlist)} \n PCA with {len(b)}/{len(subjlist)}", fontsize=12)
    ax.set(xlim=[0,1])
    
    if legend==False:
        ax.get_legend().remove()
    else: 
        ax.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()

for key in regions:
    f, ax = plt.subplots(ncols=5,nrows=1,figsize=(16,6))
    plt.subplot(151)
    plot_itHue(subjects,2,regions[key])
    plt.subplot(152)
    plot_itHue(subjects,4,regions[key])
    plt.subplot(153)
    plot_itHue(subjects,6,regions[key])
    plt.gca().set_title(key,fontsize = 20)
    plt.subplot(154)
    plot_itHue(subjects,8,regions[key])
    plt.subplot(155)
    plot_itHue(subjects,10,regions[key],legend=True)
    plt.savefig(f'./{key}_corr.png',facecolor='w')
    
    
for key in regions:
    f, ax = plt.subplots(ncols=5,nrows=1,figsize=(16,6))
    plt.subplot(151)
    plot_itHue(subjects,2,regions[key],corr=False)
    plt.subplot(152)
    plot_itHue(subjects,4,regions[key],corr=False)
    plt.subplot(153)
    plot_itHue(subjects,6,regions[key],corr=False)
    plt.gca().set_title(key,fontsize = 20)
    plt.subplot(154)
    plot_itHue(subjects,8,regions[key],corr=False)
    plt.subplot(155)
    plot_itHue(subjects,10,regions[key],legend=True,corr=False)
    plt.savefig(f'./{key}_NoCorr.png',facecolor='w')

