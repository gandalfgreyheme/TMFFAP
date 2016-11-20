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

ThetaCooc = 1 #threshold below which the co-occurances are set to 0 HYPER PARAMETER

cooc_cnt=cooc_raw>=ThetaCooc

cooc_cnt=cooc_cnt*1 #Applying delta -> 1 if cooc, 0 if otherwise

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

ThetaConsy = 0.3 
#threshold below which the normalised PMI is set to 0 HYPER PARAMETER
#ThetaConsy ->{0,1}

nPMI[nPMI<ThetaConsy]=0