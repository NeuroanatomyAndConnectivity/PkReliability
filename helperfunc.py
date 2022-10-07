import numpy as np 
import nibabel as nib 

import nilearn.plotting
import matplotlib.pyplot as plt
import mayavi
from mayavi import mlab
import ptitprince as pt
import seaborn as sn
import pandas as pd
import numba
from utils_hcpClass  import *

import networkx as nx
import gdist
import surfdist as sd 
import surfdist.analysis  

from sklearn.cluster import KMeans


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


def plot_srfs(a,b,c,d,title=False):
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
    if title !=False:
        figure.suptitle(title, fontsize=16,y=0.5)
    
    plt.tight_layout()
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

def Zscore(x):
    Z=(x-np.mean(x))/np.std(x)
    return Z

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
##### flatten a list
def flatten(l):
    return [item for sublist in l for item in sublist]

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


def prep_plotsXkernel(kernel,subjects,pca=False,corr=True):
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
    ax.set(ylabel = "Gradient Threshold")
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
    
    figure.suptitle(title, fontsize=12)


    plt.subplots_adjust(left=0.1,
                        bottom=0.5, 
                        top=0.9, 
                        wspace=-0.1, 
                        hspace=0)
    mlab.close()
    return figure
#     plt.savefig(f'{file}.png',bbox_inches='tight',facecolor='w')
#     plt.close()

def get_sensROIS(subID,hemi):
    if hemi =='R':
        V1=sub_dict[subID].RV1
        S1=sub_dict[subID].RS1
        A1=sub_dict[subID].RA1 
    elif hemi =='L':
        V1=sub_dict[subID].LV1
        S1=sub_dict[subID].LS1
        A1=sub_dict[subID].LA1

    return V1,S1,A1

def get_sens2pk(data,hemi):
    #### add a hemi argument to specify left and right ROIs
    #### set up sensory lists. we've got 9 measures per subjectper hemisphere... 
    parV1=[]
    parS1=[]
    parA1=[]
        
    tmpV1=[]
    tmpS1=[]
    tmpA1=[]
    
    mparV1=[]
    mparS1=[]
    mparA1=[]

    
    x=0 
    
    for key in data:
        
        V1,S1,A1=get_sensROIS(key,hemi)

        if np.isnan(data[key][0]).any()==False: 
            parV1.append(np.min(data[key][0][V1])) 
            parS1.append(np.min(data[key][0][S1])) 
            parA1.append(np.min(data[key][0][A1]))
        else:
            parV1.append(np.nan) 
            parS1.append(np.nan) 
            parA1.append(np.nan)
                
        if np.isnan(data[key][1]).any()==False: 
            tmpV1.append(np.min(data[key][1][V1])) 
            tmpS1.append(np.min(data[key][1][S1])) 
            tmpA1.append(np.min(data[key][1][A1]))
        else:
            tmpV1.append(np.nan) 
            tmpS1.append(np.nan) 
            tmpA1.append(np.nan)
            
            
        if np.isnan(data[key][2]).any()==False: 
            mparV1.append(np.min(data[key][2][V1])) 
            mparS1.append(np.min(data[key][2][S1])) 
            mparA1.append(np.min(data[key][2][A1]))
        else:
            mparV1.append(np.nan) 
            mparS1.append(np.nan) 
            mparA1.append(np.nan)
                
    return [parV1,parS1,parA1],[tmpV1,tmpS1,tmpA1],[mparV1,mparS1,mparA1]
            
            

            

def prep_sens2pkSubjPlots(hemi):
    """ Specify hemisphere as 'L' or 'R' """
    lParV1=[]
    lParS1=[]
    lParA1=[]
    lTmpV1=[]
    lTmpS1=[]
    lTmpA1=[]
    mParV1=[]
    mParS1=[]
    mParA1=[]

    for i in containL:
        #lateral parietal
        lParV1.append(get_sens2pk(i,hemi)[0][0])
        lParS1.append(get_sens2pk(i,hemi)[0][1])
        lParA1.append(get_sens2pk(i,hemi)[0][2])
    
        #lateral temporal
        lTmpV1.append(get_sens2pk(i,hemi)[1][0])
        lTmpS1.append(get_sens2pk(i,hemi)[1][1])
        lTmpA1.append(get_sens2pk(i,hemi)[1][2])
        
        #medial parietal
        mParV1.append(get_sens2pk(i,hemi)[2][0])
        mParS1.append(get_sens2pk(i,hemi)[2][1])
        mParA1.append(get_sens2pk(i,hemi)[2][2])
    
    to_dict=[lParV1,lParS1,lParA1,lTmpV1,lTmpS1,lTmpA1,mParV1,mParS1,mParA1]

    for i in range(len(to_dict)):
        to_dict[i]=dict(zip(sub_dict.keys(),np.vstack(to_dict[i]).T))
    return to_dict



def plot_subjectwiseDist(dic,title):
    ticks=list(range(80,100,2))
#     dic=pd.DataFrame.from_dict(dic)
    ax=sn.lineplot(data=dic,marker='o')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_xticks(range(10))
    ax.set_xticklabels(ticks)
    plt.ylabel('Distance')
    plt.xlabel('Gradient Threshold')
    plt.title(title)

def plot_pk2sens(hemi):
    out=[]
    for key in sub_dict:
        
        if hemi =='L':
            metric=sub_dict[key].LdistSens
    
            thr_dists=[]
            for thr in range(80,100,2):
                L,R=dat.get_peaks_postZones(thr)
        
                distances=[]
                for peak in range(len(L)):
                    if np.isnan(L[peak][0]).any()==True:
                        distances.append(np.nan)
                    else:
                        dist=np.nanmin(metric[L[peak]])
                        distances.append(dist)
                thr_dists.append(np.hstack(distances))
            out.append(thr_dists)
            
        if hemi =='R':
            metric=sub_dict[key].RdistSens
    
            thr_dists=[]
            for thr in range(80,100,2):
                L,R=dat.get_peaks_postZones(thr)
        
                distances=[]
                for peak in range(len(R)):
                    if np.isnan(R[peak][0]).any()==True:
                        distances.append(np.nan)
                    else:
                        dist=np.nanmin(metric[R[peak]])
                        distances.append(dist)
                thr_dists.append(np.hstack(distances))
            out.append(thr_dists)
            
    tst=dict(zip(sub_dict.keys(),out))
    latPar={}
    latTmp={}
    medPar={}
    for key in tst:
        lpar=np.vstack(tst[key])[:,0]
        latPar[key]=lpar
        ltmp=np.vstack(tst[key])[:,1]
        latTmp[key]=ltmp
        mpar=np.vstack(tst[key])[:,2]
        medPar[key]=mpar
    return latPar,latTmp,medPar


def force_df(dictionary):
    return pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dictionary.items() ]))


def get_parcels(prc):
    prc_file=nib.load(prc)
    prc=prc_file.labeltable.get_labels_as_dict()
    parcels={}
    for key in prc:
        parcels[prc[key]]=(np.where(prc_file.darrays[0].data==key)[0])
    del parcels['???']

    #### adds flexibility to other parcellations, especially those derived from cifti-spearate
    parc=parcels.copy()
    for key, value in parc.items():
        if value.size==0:
            del parcels[key]
    return parcels


def box_plot_parcelsXsub(data,plot=True):
    data=force_df(data)
#     data.drop(columns='L_Medial_wall',inplace=True)
    for key in data:
        subset=data[key].dropna()
        num=np.sum(subset==0)
        den=subset.shape
        if num==1:
#             data[key]<0.01=np.nan
            data.loc[data[key]<0.01]=np.nan
            print(key)
    
    
    re_order=list(data.mean().sort_values(ascending=True).keys())
    data=data[re_order]
#     data=data.min().to_frame().T
    labels=list(data.columns)
    melted=data.melt()
#     melted['Primary Cortex']=0
    for key in yeo_dictL:
         melted.loc[melted['variable']==key,'Primary Cortex']=yeo_dictL[key]
    
    melted.loc[melted['variable']=='L_S_calcarine','Primary Cortex']='V1'
    melted.loc[melted['variable']=='L_S_central','Primary Cortex']='S1'
    melted.loc[melted['variable']=='L_S_temporal_transverse','Primary Cortex']='A1'
    

#     custom=hcp.yeo7['rgba']
#     for key in custom:
#         custom[key]=rgb2hex(custom[key])

    custom['V1']='#FF5733'
    custom['A1']='#77F1E1'
    custom['S1']='#09ed05'
    
    if plot==True:
        fig = plt.gcf()
        fig.set_size_inches(15,10)
#         pal =sn.color_palette(custom.values())
        ax=sn.boxplot(data=melted,x='variable',y='value',hue='Primary Cortex',palette=custom,dodge=False)
        sn.despine(offset=0, trim=True);
        ax.set_xticklabels(labels,rotation = 90,size=10)
        ax.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
        plt.tight_layout()
    
    
    return data,melted,re_order
    
#### remove lone zeros 
def quitaZeros(data):
    for key in data:
        subset=data[key].dropna()
        num=np.sum(subset==0)
        den=subset.shape
        if num==1:
            data.loc[data[key]<0.01]=np.nan
    return data

def applyNetworkHue(df):
    labels=list(df.columns)
    melted=df.melt()
#     melted['Primary Cortex']=0
    for label in labels:
         melted.loc[melted['variable']==label,'Primary Cortex']=yeo_dictL[label]
    
    melted.loc[melted['variable']=='L_S_calcarine','Primary Cortex']='V1'
    melted.loc[melted['variable']=='L_S_central','Primary Cortex']='S1'
    melted.loc[melted['variable']=='L_S_temporal_transverse','Primary Cortex']='A1'
    
    return melted


def region_order(dat,thr):
    a=force_df(dat).mean().sort_values()
    a=pd.DataFrame(a)
    a.reset_index(inplace=True)
    a.reset_index(inplace=True)
    a.set_index('index',inplace=True)
    a.rename(columns={'level_0':f'order {thr}'},inplace=True) 
    a.rename(columns={0:f'dist {thr}'},inplace=True)
    a.drop('L_Medial_wall',inplace=True)
    a=a.T
    return a
    
