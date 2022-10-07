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
from utils_hcpClass import *

import networkx as nx
import gdist
import surfdist as sd 
import surfdist.analysis  

from sklearn.cluster import KMeans

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

        
        self.Raparc=f'{MNIpath}/{subj}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'
        
        self.RV1=np.where(nib.load(self.Raparc).darrays[0].data==45)[0]
        self.RS1=np.where(nib.load(self.Raparc).darrays[0].data==46)[0]
        self.RA1=np.where(nib.load(self.Raparc).darrays[0].data==75)[0]
        
        
    
        self.LZverts=get_zoneVerts(LWS)
        self.RZverts=get_zoneVerts(RWS)
    
        self.LdistSens=np.load(f'{subj}/{subj}.L.dist32K.npy')
        self.RdistSens=np.load(f'{subj}/{subj}.R.dist32K.npy')
        
        neighbours=self.neighbours
        
        if self.neighbours==None:
            pass
        else:
            self.Lneighbours=SpatialNeighbours(self.Lcoords,self.Lfaces)
            self.Rneighbours=SpatialNeighbours(self.Rcoords,self.Rfaces)
        
        
        if self.pca is None:
           
#             print('ussing diffusion maps')

            #### full gradient 
            self.grad=np.load(f'{subj}/{subj}.mapalign.diffmaps.0{kernel}mm.npy')
            self.Lgrad=self.grad[0][0:len(self.Lfill)]
            self.Lgrad=recort(self.Lgrad,self.Lfill,self.dims)
            self.Lgrad=gradientOrientation(self.Lgrad,'left',self.Laparc)


            self.Rgrad=self.grad[0][len(self.Lfill):]
            self.Rgrad=recort(self.Rgrad,self.Rfill,self.dims)
            self.Rgrad=gradientOrientation(self.Rgrad,'right',self.Raparc)

            ###### session 1 
            ### subsessions
            self.gradses1=np.load(f'{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            ######## session 2 
 
        
            self.gradses2=np.load(f'{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')
        
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
            
        else:
#             print('using PCA maps')
            ######### load PCA grads
            self.gradses1=np.load(f'{subj}/{subj}.pca.ses1.s0{kernel}mm.npy')
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
    
    def get_peaks_postZones(self,pct,visualize=False):
        #### get the thresholded gradients from session 1 and 2 
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
        Linter=[]
        for l in self.LZverts:
            #### geat each  zone's vertices
            roi=self.LZverts[l]
            ### get each sessions gradient values form the zone
            S1ZnL=np.intersect1d(roi,S1[0])
            S2ZnL=np.intersect1d(roi,S2[0])
            ### get the intersect of the peak values in the zone
            LZn_int=np.intersect1d(S1ZnL,S2ZnL)
            Linter.append(LZn_int)
        Rinter=[]
        for r in self.RZverts:
            ### do the same as above for the right
            roi=self.RZverts[r]
            S1ZnR=np.intersect1d(roi,S1[1])
            S2ZnR=np.intersect1d(roi,S2[1])
            RZn_int=np.intersect1d(S1ZnR,S2ZnR)
            Rinter.append(RZn_int)
        #### isolate just the latparietal, latTmp, and medparietal peaks
        posteriorZones=[1,4,6]
        Linter=[Linter[i] for i in posteriorZones]
        Rinter=[Rinter[i] for i in posteriorZones]

 
        if len(Linter[0])==0 or len(Rinter[0])==0:
#             print(f'no lateral parietal -- at threshold {pct}')
            Linter[0]=np.nan
            Rinter[0]=np.nan
#             out=np.asarray([np.nan,np.nan,np.nan],dtype=object)
#             return out,out

        if len(Linter[1])==0 or len(Rinter[1])==0:
#             print(f'no lateral temporal -- at threshold {pct}')
            Linter[1]=np.nan
            Rinter[1]=np.nan
#             out=np.asarray([np.nan,np.nan,np.nan],dtype=object)
#             return out,out

        if len(Linter[2])==0 or len(Rinter[2])==0:
#             print(f'no medial parietal -- at threshold {pct}')
            Linter[2]=np.nan
            Rinter[2]=np.nan
#             out=np.asarray([np.nan,np.nan,np.nan],dtype=object)
#             return out,out
       
        Lpks=[]
        for zone in Linter:
            if np.isnan(zone).any()==True:
                Lpks.append(np.asarray([np.nan]))
            else:
                dat={}
                for i in zone:
                    dat[i]=self.Lneighbours[i]
                G=nx.Graph(dat)
                largest_cc = max(nx.connected_components(G), key=len,default=None)
                Lpks.append(np.asarray(list(largest_cc)))
        
        
        
        Rpks=[]
        for zone in Rinter:
            if np.isnan(zone).any()==True:
                Rpks.append(np.asarray([np.nan]))
            else:
                dat={}
                for i in zone:
                    dat[i]=self.Rneighbours[i]
                G=nx.Graph(dat)
                largest_cc = max(nx.connected_components(G), key=len,default=None)
                Rpks.append(np.asarray(list(largest_cc)))

        
        if visualize != False:
#             print('muajaja we viz this fucker')
            
            L=np.zeros(self.dims)
            R=np.zeros(self.dims)
            
            x=0
            for i in range(len(Lpks)):
                if np.isnan(Lpks[i]).any()==True or np.isnan(Rpks[i]).any()==True:
                    pass
                else:
                    L[Lpks[x]]=x+1
                    R[Rpks[x]]=x+1
                x=x+1
            
            a,b=oh_mayavi(self.Linflated,L,'PuBuGn',False)
            mlab.clf()
            d,c=oh_mayavi(self.Rinflated,R,'PuBuGn',False)

            plot_srfs(a,b,c,d,f'{self.subj} Threshold {pct}')
        
        
        
        
        
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
        a,b=oh_mayavi(self.Linflated,self.LdistSens,'CMRmap',clrbar=False)
        mlab.clf()
        c,d=oh_mayavi(self.Rinflated,self.RdistSens,'CMRmap',clrbar=False)
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
        
    def get_fullThrIntersect(self,pct):
        L=np.zeros(self.dims)
        R=np.zeros(self.dims)
        
        topXSes1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        topXSes2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
        
        
        L1=topXSes1[0]
        L2=topXSes2[0]
        Linter=np.intersect1d(L1,L2)

        
        R1=topXSes1[1]
        R2=topXSes2[1]
        Rinter=np.intersect1d(R1,R2)
        
        return Linter,Rinter
        
    
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
        
        
