# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:10:39 2016

Logical Item Set Mining (LISM) V 0.0

LISM Implementation in Python. 
Read more here: http://cvit.iiit.ac.in/papers/Chandrashekar2012Logical.pdf


@author: saurabh.choudhary
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os

#Setting Hyperparameters

#Threshold below which the co-occurances are set to 0
#ThetaCooc>=1; int
ThetaCooc = 1

#Threshold below which the normalised PMI is set to 0
#ThetaConsy ->{0,1}
ThetaConsy = 0.3

#Number of iterations for denoising
nIter=10

#Stage 0 - Get Data

path = r"D:\Data Experiments\VH Verbatim Analysis"

os.chdir(path)

text=os.path.join(path, "TDM_test.csv")

tDF=pd.read_csv(text)

vec=CountVectorizer(stop_words='english',binary=True)

tdm=vec.fit_transform(tDF['Comments'])

tdm=tdm.toarray()

words=vec.get_feature_names()

words=np.array(words)

#nTDM = pd.DataFrame(tdm, columns=words) #named TDM

#Stage 1: LISM counting
#      1.1: Cooccurance Counts {psi (alpha, beta)}
cooc_raw = np.dot(tdm.transpose(),tdm) #Calculate Co-occurance Matrix

np.fill_diagonal(cooc_raw, 0) #make the cooc matrix diagnol 0 P1 != P2

cooc_zero=cooc_raw>=ThetaCooc

cooc_zero=cooc_zero*1 #Applying delta -> 1 if cooc, 0 if otherwise

cooc_cnt=cooc_zero

for i in range(0,nIter):
    #      1.2: Margianal Counts {psi (alpha)}
    
    marginal_cnt = np.sum(cooc_cnt, axis=1)
    
    #      1.3: Total Counts {psi0}
    
    total_cnt=0.5*np.sum(marginal_cnt)
    
    #Calculate Co Oc and Marginal probabilities
    
    cooc_prob=cooc_cnt/total_cnt #P(a,b)
    
    marginal_prob=marginal_cnt/total_cnt #P(a)
    
    #Stage 2: LISM consistency
    
    #Calculate pointwise mutual information = max{0,ln(P(a,b)/(P(a)*P(b))}
    
    PMI = cooc_prob/np.outer(marginal_prob,marginal_prob) 
    # outer product produces an MxM matrix for an M dimensional vector giving us
    # a vectorised implementation of P(a)*P(b)
    
    PMI=np.log(PMI)
    
    PMI[PMI<=0]=0
    
    #Calcluate normalised PMI = PMI/(-ln(P(a,b)))
    
    ln_cooc_prob = -np.log(cooc_prob)
    
    nPMI=np.divide(PMI,ln_cooc_prob)
    
    nPMI[np.isnan(nPMI)==True]=0
    
    nPMI[nPMI<ThetaConsy]=0
       
    nPMImask=nPMI>0
    
    cooc_count_t0=np.sum(cooc_cnt)    
    
    cooc_cnt=np.multiply(cooc_zero,nPMImask)
    
    quality=np.sum(np.multiply(cooc_prob,nPMI))
    
    print("Quality = {}".format(quality))
    
    if(np.sum(cooc_cnt)==cooc_count_t0):
        print("Iterations taken = {}".format(i+1))
        print("Number of latent structures found = {}".format(np.sum(cooc_cnt)/2))        
        break
    elif(i==9):
        print("Did not converge in {} iterations".format(nIter))