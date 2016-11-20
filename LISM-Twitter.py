# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:03:04 2016

@author: saurabh.choudhary
https://github.com/gandalfgreyheme/TMFFAP/
"""

import tweepy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import networkx as nx
import matplotlib.pylab as plt
import itertools as it
import re

#Global parameters to cycle colors

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

#Stage 0: Pull data from Twitter

path = r"C:\Analytics"

os.chdir(path)

consumer_key = 's8MmU4HOs8u142rNmj7xNNcqD'
consumer_secret = 'fGP4IZ0s10S3eiLcO1MAdTFVPL0UXUJ4vyojCiFTSHkn4xdPVu'

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
 
api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)
 
if (not api):
    print ("Can't Authenticate")
    #sys.exit(-1)
    
searchQuery = '#DeMonetisation'  # this is what we're searching for
maxTweets = 10000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits

tweetCount = 0
new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en")
txt=[]
while len(txt)<=maxTweets:
    for tweet in new_tweets:
        if not (tweet.text).startswith('RT'):    
            txt.append(tweet.text)

print("Downladed {} tweets!".format(len(txt)))

#strip special characters

tDF=pd.DataFrame(data=txt,columns=["Tweet"])

tDF['Tweet'] = tDF['Tweet'].map(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))

#vec=CountVectorizer(stop_words='english',binary=True)

vec=CountVectorizer(binary=True)

tdm=vec.fit_transform(tDF['Tweet'])

tdm=tdm.toarray()

words=vec.get_feature_names()

words=np.array(words)

print("Tweets vectorised, {} words found.".format(len(words)))

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

for clique in cliques:
    print "Clique to appear: ",clique
    H = g.subgraph(clique)
    col=colors.next()    
    nx.draw_networkx(H, node_colors=col,with_Lables=True)
    plt.show()    
    plt.clf()