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
import networkx as nx
import matplotlib.pylab as plt
import itertools as it
import re

#Draw circle around cliques
#http://stackoverflow.com/questions/9213797/graphviz-drawing-maximal-cliques
def draw_circle_around_clique(clique,coords):
    dist=0
    temp_dist=0
    center=[0 for i in range(2)]
    color=colors.next()
    for a in clique:
        for b in clique:
            temp_dist=(coords[a][0]-coords[b][0])**2+(coords[a][1]-coords[b][1])**2
            if temp_dist>dist:
                dist=temp_dist
                for i in range(2):
                    center[i]=(coords[a][i]+coords[b][i])/2
    rad=dist**0.5/2
    cir = plt.Circle((center[0],center[1]),   radius=rad*1.3,fill=False,color=color,hatch=hatches.next())
    plt.gca().add_patch(cir)
    plt.axis('scaled')
    # return color of the circle, to use it as the color for vertices of the cliques
    return color

global colors, hatches
colors=it.cycle('bgrcmyk')# blue, green, red, ...
hatches=it.cycle('/\|-+*')

#Setting Hyperparameters

#Threshold below which the co-occurances are set to 0
#ThetaCooc>=1; int
ThetaCooc = 3

#Threshold below which the normalised PMI is set to 0
#ThetaConsy ->{0,1}
ThetaConsy = 0.3

#Number of iterations for denoising
nIter=10

#Stage 0 - Get Data

path = r"D:\Data Experiments\VH Verbatim Analysis"

os.chdir(path)

text=os.path.join(path, "TDM_test_2.csv")

tDF=pd.read_csv(text)

#strip special characters
tDF['Comments'] = tDF['Comments'].map(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))

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
        
#Finding maximal cliques
        
#retain only the upper triangle of the cooc matrix (because its symmetric)

cooc_upper=np.triu(cooc_cnt,k=0)

#create graph object

g=nx.Graph()

g.add_nodes_from(words) #add the words we identified as nodes
    
for i in range(0,cooc_upper.shape[0]):
    for j in range(0,cooc_upper.shape[1]):    
        if cooc_upper[i,j]==1: # add an edge only if both values are provided
            g.add_edge(words[i],words[j])

# Remove nodes with no edges
degree=g.degree()
for n in g.nodes():
    if degree[n]==0:
        g.remove_node(n)
        
#Find maximal cliques and visualise
        
coords=nx.spring_layout(g)

# remove "len(clique)>2" if you're interested in maxcliques with 2 edges
cliques=[clique for clique in nx.find_cliques(g) if len(clique)>2]

#draw the graph
#nx.draw(g,pos=coords)
for clique in cliques:
    print "Clique to appear: ",clique
    H = g.subgraph(clique)
    nx.draw_networkx(H, with_Lables=True)
    plt.show()    
    plt.clf()

#nx.draw_networkx(g,pos=coords,nodelist=clique,\
#node_color=draw_circle_around_clique(clique,coords), with_Lables=True)

#plt.savefig("network.jpg")

plt.show()