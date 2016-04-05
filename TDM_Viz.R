#########################################
####  BLOCK 1 Data Prep and Library   ###
#########################################

library(tm)

#Import verbatim data - copy the verbatim and paste in a single row of a CSV file, do a remove duplicates function in excel before saving

setwd("D:/R_Out")

MH_Extract<-read.csv("Insta_Solly_noallen.csv",header=TRUE) #import the full CSV file of the data import

#posboost = read.table("pos_boosters.txt",header=FALSE)

#negboost = read.table("neg_boosters.txt",header=FALSE)

MH_Extract$Comments[MH_Extract$Comments==""] <- NA

MH_Extract <- MH_Extract[is.na(MH_Extract$Comments)==0,]

#Only include statements containing "WORD" into the analysis

#MH_Extract<-MH_Extract[grepl('festive',MH_Extract$Comments),]

#write.csv(MH_Extract, "Extract_Only.csv")

# Exclusion statement ends

rawVerbatim <- data.frame(MH_Extract$Comments)

myCorpus <- Corpus(DataframeSource(rawVerbatim))

myCorpus <- tm_map(myCorpus, stripWhitespace)  #strip extra spaces

myCorpus <- tm_map(myCorpus, tolower)  #converts all test to lower case

myCorpus <- tm_map(myCorpus, removeWords, stopwords('english'))  #removes all stop words

#myCorpus <- tm_map(myCorpus, removeNumbers)  #removes all numbers

myCorpus <- tm_map(myCorpus, removePunctuation)   #removes punctuation

#myCorpus <- tm_map(myCorpus, stemDocument)   #stems words to word roots e.g. manager, managed, managing -> manag

myCorpus <- tm_map(myCorpus, PlainTextDocument)


##################################
####  BLOCK 1 Ends Here        ###
##################################


##################################
#########  BLOCK 2 Wordcloud   ###
##################################


myTDM <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))#, weighting = weightTfIdf))

#myDTM <- DocumentTermMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))

# tdm <- as.matrix(myTDM)

#dtm <- as.matrix(myDTM)

sTDM<-removeSparseTerms(myTDM, 0.99)
smTDM<-as.matrix(sTDM)

#(freq.terms <- findFreqTerms(myTDM, lowfreq= 2))

##get the names of the 10 words that correlate the highest with query
#words <- rownames(findAssocs(myDTM, "rayban", .11))
#find <- colnames(dtm) %in% words
#corr <- cor(dtm[,find])
#plot heatmap of correlations
#library(corrplot)
#corrplot(corr, type = "upper")



###Show terms frequencies with histogram
# can see the Zipf's law !
#term.freq <- rowSums(tdm)
#term.freq <- subset(term.freq, term.freq>=2)
#word_freqs = sort(term.freq, decreasing=FALSE) 
#vocab <- names(word_freqs)
# create a data frame with words and their frequencies
#df = data.frame(terms=vocab, freq=word_freqs)


library(ggplot2)
#df$terms <- factor( df$terms, levels=unique(as.character(df$terms)) )
#ggplot(df, aes(terms,freq)) + geom_bar(stat= "identity") + scale_x_discrete(name="Terms", labels=df$terms) + xlab("Terms") + ylab("Freq") + coord_flip()

######### TERMS * TERMS Matrix (Graph) #######
# transform into a term-term adjacency matrix
btdm<-smTDM
btdm[btdm>=1] <- 1
xtdm <- btdm %*% t(btdm)
#### Create a graph from it
library(igraph)
#build a graph from the above matrix
#g <- graph.adjacency(xtdm, weighted=T, mode="undirected")
# remove loops
#g <- simplify(g)
### Visualize it
#plot.igraph(g, layout=layout.fruchterman.reingold(g, niter=1000, area=100*vcount(g)^2))
#mtext("Terms Co-occurrences", side=1)


G<-graph.adjacency(xtdm, mode=c("undirected"))               # convert adjacency matrix to an igraph object
cent<-data.frame(bet=betweenness(G),eig=evcent(G)$vector) # calculate betweeness & eigenvector centrality 
res<-as.vector(lm(eig~bet,data=cent)$residuals)           # calculate residuals
cent<-transform(cent,res=res)                             # add to centrality data set
write.csv(cent,"r_keyactorcentrality.csv")                # save in project folder

#plot(G, layout = layout.fruchterman.reingold)             # network visualization

# create vertex names and scale by centrality
#plot(G, layout = layout.fruchterman.reingold, vertex.size = 20*evcent(G)$vector, vertex.label = as.factor(rownames(cent)), main = 'Network Visualization in R')

# key actor analysis - plot eigenvector centrality vs. betweeness
# and scale by residuals from regression: eig~bet

#library(ggplot2)

#p<-ggplot(cent,aes(x=bet,y=eig,label=rownames(cent),colour=res, size=abs(res)))+xlab("Betweenness Centrality")+ylab("Eigenvector Centrality")
#pdf('key_actor_analysis.pdf')
#p+geom_text()+opts(title="Key Actor Analysis")   