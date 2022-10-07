import numpy as np 
import nibabel as nib 

import nilearn.plotting
import matplotlib.pyplot as plt
#import mayavi
#from mayavi import mlab
#import ptitprince as pt
import seaborn as sn
import pandas as pd

from utils import *

#import networkx as nx
#import gdist
#import surfdist as sd 
#import surfdist.analysis  

from sklearn.cluster import KMeans


LWS=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/LWS.28.max.label.gii').darrays[0].data
RWS=nib.load('/well/margulies/users/mnk884/PkReliability/watershed_templates/RWS.28.max.label.gii').darrays[0].data


def recort(X,fill,dims):
    out=np.zeros(dims)
    out[fill]=X
    return out

def recort_bin(X,fill,dims):
    out=np.zeros(dims)
    mini=np.zeros(len(fill))
    mini[X]=1
    out[fill]=mini
    return out
    

def binit(X):
    pct=np.percentile(X,[10,20,30,40,50,60,70,80,90])
    pct=np.digitize(X,pct)
    return pct+1 

def binit20(X):
    pct=np.percentile(X,[20,40,60,80])
    pct=np.digitize(X,pct)
    return pct+1 


def get_zoneVerts(WS):
    zoneverts={}
    for i in range(1,np.max(WS)):
        zoneverts.update({f'zone{i}':np.where(WS==i)[0]})
    return zoneverts

def oh_mayavi(surf,stat,cmap,clrbar=True):
    """surface, statmap, colormap"""
    ##### parse the gifti 
    anat=nib.load(surf)
    coords=anat.darrays[0].data
    x=coords[:,0]
    y=coords[:,1]
    z=coords[:,2]
    triangles=anat.darrays[1].data
    ##### if subcortical mask provided use it

#         print('masking out subcortex')
#     sub_cort=nilearn.surface.load_surf_data(args[0])
#     stat[sub_cort]=float('NaN')

    
    
    ### start mayavi 
    
    maya=mlab.triangular_mesh(x,y,z,triangles,scalars=stat,colormap=cmap)
    
    mlab.view(azimuth=0, elevation=-90)
    f = mlab.gcf()
    cam = f.scene.camera
    cam.zoom(1.)
    if clrbar==True:
        cb=mlab.colorbar(orientation='vertical', nb_labels=3,label_fmt='%.2f')
        cb.label_text_property.color =(0,0,0)
    else: 
        pass

    mlab.draw()
    

    img1=mlab.screenshot(figure=maya,mode='rgba',antialiased=True)
    mlab.view(azimuth=0, elevation=90)
    mlab.figure(bgcolor=(0, 0, 0))
    ### clear figure
    mayavi.mlab.clf()
    
    f = mlab.gcf()
    cam = f.scene.camera
    cam.zoom(1.1)
    mlab.draw()
    img2=mlab.screenshot(figure=maya,mode='rgba',antialiased=True)
    ### clear figure
    mayavi.mlab.clf()
    mlab.clf()
    mlab.close()
    return img1,img2


def plot_srfs(a,b,c,d):
    figure=plt.figure(figsize=(6, 8), dpi=180)
    plt.subplot(2,2,1)
    plt.imshow(a)
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(b)
    plt.axis('off')
    mlab.clf()
    plt.subplot(2,2,3)
    plt.imshow(c)
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(d)
    plt.axis('off')
    

    plt.subplots_adjust(left=0.1,
                        bottom=0.5, 
                        top=0.9, 
                        wspace=0, 
                        hspace=0)
#     plt.savefig(f'{file}.png',bbox_inches='tight',facecolor='w')
#     plt.close()


def SpatialNeighbours(coords,faces):
    #### gives the same output as mris_convert -v but directly into a python structure
    neighbours={}
    for i in range(len(coords)):
        a=np.unique(faces[np.where(faces==i)[0]])
        a=a[a!=i]
        neighbours[i]=a.tolist()
    return neighbours


def dice_it(A,B):
    
    num=2*(len(np.intersect1d(A,B)))
    den=len(A)+len(B)
    
    if den ==0:
        return np.nan
    else:
        return num/den

def jaccard_it(A,B):
    num=len(np.intersect1d(A,B))
    den=len(np.union1d(A,B))
    
    if den ==0:
        return 0
    else:
        return num/den

def gradientOrientation(grad,hemi,aparc):
    """Determine the orientation of the gradients, and also return whether valid for continued study or not"""
    grad=grad #nib.load(grad).agg_data()
    if hemi=='left':
        labels=nib.load(aparc).agg_data()
#       print('getting gradient orientation from left hemisphere')
    else:
        labels=nib.load(aparc).agg_data()
#       print('getting gradient orientation from right hemisphere')
    calc=np.where(labels==45)[0]
    ctr=np.where(labels==46)[0]
    if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
#       print('Canonical Orientation DMN at apex')
        return grad,True
    elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
#       print(f'REMOVE {subj} FROM STUDY')
        return grad,False
    elif np.sum(grad[calc])>0 and np.sum(grad[ctr])<0:
#       print(f'REMOVE {subj} FROM STUDY')
        return grad,False
    else:
#       print('flipping gradient orientation for peak detection')
        return grad *-1,True


def prep_plotting(subj,kernel,sim='dice',pca=False):
    
    thr=[50,55,60,65,70,75,80,85,90,95]
    
    ctx_metric=[]
    zone_metricsL=[]
    zone_metricsR=[]
    
    if pca == False:
        gr=hcp_subj(subj,kernel)
        if gr.Lgradses1[1] == False or gr.Lgradses2[1] == False or gr.Rgradses1[1] ==False or gr.Rgradses2[1] ==False:
#             print(f'subject {gr.subj} Diffusion Mapping is not valid at smoothing kernel {kernel} ')
            return [gr.subj,kernel],[gr.subj,kernel],[gr.subj,kernel]
        else:
            if sim=='dice':
                for t in thr:
                    ctx_metric.append(gr.dice_Ses12(t))
                    zone_metricsL.append(gr.ZoneDice_Ses12(t)[0])
                    zone_metricsR.append(gr.ZoneDice_Ses12(t)[1])
            else: 
                for t in thr:
                    gr.Jaccard_Ses12(t)
                    ctx_metric.append(gr.Jaccard_Ses12(t))
                    zone_metricsL.append(gr.ZoneDice_Ses12(t)[0])
                    zone_metricsR.append(gr.ZoneDice_Ses12(t)[1])
                    
        
    else:
        gr=hcp_subj(subj,kernel,pca=True)
        if gr.Lgradses1[1] == False or gr.Lgradses2[1] == False or gr.Rgradses1[1] ==False or gr.Rgradses2[1] ==False:
#             print(f'subject {gr.subj} PCA is not valid at smoothing kernel {kernel} ')
            return [gr.subj,kernel],[gr.subj,kernel],[gr.subj,kernel]
        else:
            if sim=='dice':
                for t in thr:
                    ctx_metric.append(gr.dice_Ses12(t))
                    zone_metricsL.append(gr.ZoneDice_Ses12(t)[0])
                    zone_metricsR.append(gr.ZoneDice_Ses12(t)[1])
            else: 
                for t in thr:
                    gr.Jaccard_Ses12(t)
                    ctx_metric.append(gr.Jaccard_Ses12(t))
                    
                    zone_metricsL.append(gr.ZoneDice_Ses12(t)[0])
                    zone_metricsR.append(gr.ZoneDice_Ses12(t)[1])
                    
    return np.vstack(ctx_metric),np.vstack(zone_metricsL),np.vstack(zone_metricsR)


def prep_plotsXkernel(kernel,pca=False,corr=True):
    ### set up outputs
    Lhemi=[]
    Rhemi=[]
    LlatPar=[]
    LTmp=[]
    LmedPar=[]
    RlatPar=[]
    RTmp=[]
    RmedPar=[]
    
    nogo=[]
    
    ### loop through subjjects for specified kernel
    for subj in subjects:
        a,b,c=prep_plotting(subj,kernel,pca=pca)
        if len(a)>2:
            Lhemi.append(a[:,0])
            Rhemi.append(a[:,1])
        
            LlatPar.append(b[:,0])
            LTmp.append(b[:,1])
            LmedPar.append(b[:,2])
        
            RlatPar.append(c[:,0])
            RTmp.append(c[:,1])
            RmedPar.append(c[:,2])
        else:
            nogo.append(a[0])
    Lhemi=np.vstack(Lhemi)
    Rhemi=np.vstack(Rhemi)
    
    LlatPar=np.vstack(LlatPar)
    LTmp=np.vstack(LTmp)
    LmedPar=np.vstack(LmedPar)
    
    RlatPar=np.vstack(LlatPar)
    RTmp=np.vstack(LTmp)
    RmedPar=np.vstack(LmedPar)
    
    if corr == True:
        corr=(len(Lhemi)/len(subjects))
        return   (Lhemi*corr),(Rhemi*corr),(LlatPar*corr),(LTmp*corr),(LmedPar*corr),(RlatPar*corr),(RTmp*corr),(RmedPar*corr),nogo
    elif corr == False:
        return  Lhemi,Rhemi,LlatPar,LTmp,LmedPar,RlatPar,RTmp,RmedPar,nogo
    
def plot_itHue(k,rgn,legend=False,corr=True):
    """where k = kernel of 2,4,6,8,10 and rgn indexes the output of prep"""
    if corr==True:
        a=prep_plotsXkernel(k)
        b=prep_plotsXkernel(k,pca=True)
    elif corr==False:
        a=prep_plotsXkernel(k,corr=False)
        b=prep_plotsXkernel(k,pca=True,corr=False)

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
    ax.set(ylabel = "")
    plt.xlabel(f"{k}mm Smoothing.\n Dmap with {len(a)}/20 \n PCA with {len(b)}/20 subjects", fontsize=12)
    ax.set(xlim=[0,1])
    
    if legend==False:
        ax.get_legend().remove()
    else: 
        ax.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()

def plot_srfs_dice(a,b,c,d,title,diceL,diceR):
    figure=plt.figure(figsize=(2,4), dpi=180)
    plt.subplot(2,2,1)
    plt.imshow(a)
    plt.title(diceL,fontsize=6)
    plt.axis('off')
    plt.subplot(2,2,2)
    
    plt.imshow(b)
    plt.axis('off')
    mlab.clf()
    plt.subplot(2,2,3)
    plt.imshow(c)
    plt.title(diceR,fontsize=6)
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(d)
    plt.axis('off')
    
    figure.suptitle(title, fontsize=6)


    plt.subplots_adjust(left=0.1,
                        bottom=0.5, 
                        top=0.9, 
                        wspace=-0.1, 
                        hspace=0)
    mlab.close()
    return figure
#     plt.savefig(f'{file}.png',bbox_inches='tight',facecolor='w')
#     plt.close()


class hcp_subj:

    def __init__(self,subj,kernel,pca=None):
        
        self.subj=subj
        cluster_path='/well/win-hcp/HCP-YA/subjectsAll'

        GradRepo='/well/margulies/projects/data/hcpGrads'
        subjdir=f'{cluster_path}/{subj}'
        anatdirNat=f'{subjdir}/T1w/Native/'
        anatdir32=f'{subjdir}/T1w/fsaverage_LR32k/'

        MNIdir=f'{subjdir}/MNINonLinear/fsaverage_LR32k'
        self.info=np.load(f'{GradRepo}/{subj}/{subj}.cifti.info.npy',allow_pickle=True).item()

        
        self.dims=self.info['lnverts']
        self.Lfill=self.info['lIDX']
        self.Rfill=self.info['rIDX']
        self.pca=pca
        
        self.Lsrf=f'{anatdir32}/{subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.Lcoords=nib.load(self.Lsrf).darrays[0].data
        self.Lfaces=nib.load(self.Lsrf).darrays[1].data
        
        self.Linflated=f'{anatdir32}/{subj}.L.inflated_MSMAll.32k_fs_LR.surf.gii'
        
        self.Rsrf=f'{anatdir32}/{subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.Rcoords=nib.load(self.Rsrf).darrays[0].data
        self.Rfaces=nib.load(self.Rsrf).darrays[1].data


        self.Laparc=f'{MNIdir}/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'
        self.Rinflated=f'{anatdir32}/{subj}.R.inflated_MSMAll.32k_fs_LR.surf.gii'

        
        self.Raparc=f'{MNIdir}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'
        
    
        self.LZverts=get_zoneVerts(LWS)
        self.RZverts=get_zoneVerts(RWS)
    
        # self.LdistSens=np.load(f'{subj}/{subj}.L.dist32K.npy')
        # self.RdistSens=np.load(f'{subj}/{subj}.R.dist32K.npy')
        
        

        if self.pca is None:
            ########## session 1 
#             print('ussing diffusion maps')
            self.gradses1=np.load(f'{GradRepo}/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
        

        
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            ######## session 2 
 
        
            self.gradses2=np.load(f'{GradRepo}/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')
        
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
            
        else:
#             print('using PCA maps')
            ######### load PCA grads
            self.gradses1=np.load(f'{GradRepo}/{subj}/{subj}.pca.ses1.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            self.gradses2=np.load(f'{subj}/{subj}.pca.ses2.s0{kernel}mm.npy')   
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
        
        
        
    
    def print_subj(self):
        print(self.subj)
    
    
    ### extract the vertices associated with each zone
    def zoning(self,Lgrad,Rgrad):
        Lg_zone=[]
        for key in self.LZverts:
            Lroi=self.LZverts[key]
            Lg_zone.append(Lgrad[0][Lroi])
        
        Rg_zone=[]

        for key in self.RZverts:
            Rroi=self.RZverts[key]
            Rg_zone.append(Rgrad[0][Rroi])
        return Lg_zone,Rg_zone

    
    def extract_topX(self,Left,Right,pct):
        """extract the top X percent instead of binning"""
        
        #######################################################
        #######################################################
        #######################################################
        # ADD CHECK HERE TO MAKE SURE IT IS A VALID GRADIENT WITH THE L[1] AND R[1]
        # THESE ARE SAVED AND HERE TO BE USED. DON'T WASTE THAT CHECK
        
        
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
    
    def ZoneDice_Ses12(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
        Ldx=[]
        for l in self.LZverts:
            roi=self.LZverts[l]
            S1ZnL=np.intersect1d(roi,S1[0])
            S2ZnL=np.intersect1d(roi,S2[0])
            diceLZn=dice_it(S1ZnL,S2ZnL)
            Ldx.append(diceLZn)
        Rdx=[]
        for r in self.RZverts:
            roi=self.RZverts[r]
            S1ZnR=np.intersect1d(roi,S1[1])
            S2ZnR=np.intersect1d(roi,S2[1])
            diceRZn=dice_it(S1ZnR,S2ZnR)
            Rdx.append(diceRZn)
        
        posteriorZones=[1,4,6]
        Ldx=[Ldx[i] for i in posteriorZones]
        Rdx=[Rdx[i] for i in posteriorZones]
        
        return np.asarray(Ldx),np.asarray(Rdx)
    
    def get_peaks_postZones(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
        Linter=[]
        for l in self.LZverts:
            roi=self.LZverts[l]
            S1ZnL=np.intersect1d(roi,S1[0])
            S2ZnL=np.intersect1d(roi,S2[0])
            LZn_int=np.intersect1d(S1ZnL,S2ZnL)
            Linter.append(LZn_int)
        Rinter=[]
        for r in self.RZverts:
            roi=self.RZverts[r]
            S1ZnR=np.intersect1d(roi,S1[1])
            S2ZnR=np.intersect1d(roi,S2[1])
            RZn_int=np.intersect1d(S1ZnR,S2ZnR)
            Rinter.append(RZn_int)
        
        posteriorZones=[1,4,6]
        Linter=[Linter[i] for i in posteriorZones]
        Rinter=[Rinter[i] for i in posteriorZones]
        

        if len(Linter[-1])==0 or len(Rinter[-1])==0:
            print('no medial parietal -- discard or use a lower threshold')
            sys.exit(0)
        
        
        
        neigh=SpatialNeighbours(self.Lcoords,self.Lfaces)
        
        
        Lpks=[]
        for zone in Linter:
            dat={}
            for i in zone:
                dat[i]=neigh[i]
            G=nx.Graph(dat)
            largest_cc = max(nx.connected_components(G), key=len)
            Lpks.append(np.asarray(list(largest_cc)))
        
        Rpks=[]
        for zone in Rinter:
            dat={}
            for i in zone:
                dat[i]=neigh[i]
            G=nx.Graph(dat)
            largest_cc = max(nx.connected_components(G), key=len)
            Rpks.append(np.asarray(list(largest_cc)))
            
        
        
        
        
        
        
        return np.asarray(Lpks,dtype=object),np.asarray(Rpks,dtype=object)
    
    
    
    def dice_Ses12(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
      
        diceL=dice_it(S1[0],S2[0])
        diceR=dice_it(S1[1],S2[1])
        
        
        return np.asarray([diceL,diceR])
    
    
    ############ implement the jaccard metric too
    def ZoneJaccard_Ses12(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
        Ldx=[]
        for l in self.LZverts:
            roi=self.LZverts[l]
            S1ZnL=np.intersect1d(roi,S1[0])
            S2ZnL=np.intersect1d(roi,S2[0])
            diceLZn=jaccard_it(S1ZnL,S2ZnL)
            Ldx.append(diceLZn)
        Rdx=[]
        for r in self.RZverts:
            roi=self.RZverts[r]
            S1ZnR=np.intersect1d(roi,S1[1])
            S2ZnR=np.intersect1d(roi,S2[1])
            diceRZn=jaccard_it(S1ZnR,S2ZnR)
            Rdx.append(diceRZn)
        
        posteriorZones=[1,4,6]
        Ldx=[Ldx[i] for i in posteriorZones]
        Rdx=[Rdx[i] for i in posteriorZones]
        
        return np.asarray(Ldx),np.asarray(Rdx)
    
    
    
    def Jaccard_Ses12(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
      
        diceL=jaccard_it(S1[0],S2[0])
        diceR=jaccard_it(S1[1],S2[1])
        
        
        return np.asarray([diceL,diceR])
    
    
    
  
        
    def plot_gradses1(self):
        a,b=oh_mayavi(self.Lsrf,self.Lgradses1[0],'CMRmap')
        mlab.clf()
        c,d=oh_mayavi(self.Rsrf,self.Rgradses1[0],'CMRmap')
        plot_srfs(a,b,d,c)
        
    def plot_gradses2(self):
        a,b=oh_mayavi(self.Lsrf,self.Lgradses2[0],'CMRmap')
        mlab.clf()
        c,d=oh_mayavi(self.Rsrf,self.Rgradses2[0],'CMRmap')
        plot_srfs(a,b,d,c)
        
        
    def plot_sensDist(self):
        a,b=oh_mayavi(self.Lsrf,self.LdistSens,'CMRmap')
        mlab.clf()
        c,d=oh_mayavi(self.Rsrf,self.RdistSens,'CMRmap')
        plot_srfs(a,b,d,c)
        
    def plot_topX(self,Left,Right,pct):
        L=np.zeros(self.dims)
        R=np.zeros(self.dims)
        
        topX=self.extract_topX(Left,Right,pct)
        L[topX[0]]=1
        R[topX[1]]=1
        
        a,b=oh_mayavi(self.Lsrf,L,'terrain',False)
        mlab.clf()
        c,d=oh_mayavi(self.Rsrf,R,'terrain',False)
        plot_srfs(a,b,d,c)
        
        
    def plot_ThrIntersectCortex(self,pct):
        
        L=np.zeros(self.dims)
        R=np.zeros(self.dims)
        
        topXSes1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        topXSes2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)


        
        
        
        L1=topXSes1[0]
        L2=topXSes2[0]
        Linter=np.intersect1d(L1,L2)
        Lunion=np.union1d(L1,L2)
        
#         print(f'Left Dice is {dice_it(L1,L2)}')
#         print(f'Left Jaccard is {jaccard_it(L1,L2)}')
        
        L[Lunion]=5
        L[Linter]=10
        
        R1=topXSes1[1]
        R2=topXSes2[1]
        Rinter=np.intersect1d(R1,R2)
        Runion=np.union1d(R1,R2)
        
#         print(f'Right Dice is {dice_it(R1,R2)}')
#         print(f'Right Jaccard is {jaccard_it(R1,R2)}')
        
        R[Runion]=5
        R[Rinter]=10
        
        a,b=oh_mayavi(self.Linflated,L,'PuBuGn',False)
        mlab.clf()
        d,c=oh_mayavi(self.Rinflated,R,'PuBuGn',False)
        
#         a,b=oh_mayavi(self.Lsrf,L,'PuBuGn',False)
#         c,d=oh_mayavi(self.Rsrf,R,'PuBuGn',False)
        plot_srfs_dice(a,b,c,d,f'{self.subj} Threshold {pct}',f'Left Dice is {dice_it(L1,L2):.2f}',f'Right Dice is {dice_it(R1,R2):.2f}')
        
        
