#########################################
####  BLOCK 1 Data Prep and Library   ###
#########################################

library(RWeka)

library(tm)

library(SnowballC)

library(wordcloud)

library(slam)

library(qdap)

library(plyr)

library(stringr)

#Import verbatim data - copy the verbatim and paste in a single row of a CSV file, do a remove duplicates function in excel before saving

setwd("D:/R_Out")

MH_Extract<-read.csv("MH_extract.CSV",header=TRUE) #import the full CSV file of the data import

#posboost = read.table("pos_boosters.txt",header=FALSE)

#negboost = read.table("neg_boosters.txt",header=FALSE)

MH_Extract$Comments[MH_Extract$Comments==""] <- NA

MH_Extract <- MH_Extract[is.na(MH_Extract$Comments)==0,]

#Only include statements containing "WORD" into the analysis

# MH_Extract<-MH_Extract[grepl('festive',MH_Extract$Comments),]

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

#create WordCloud

#wordcloud(myCorpus, scale=c(2.5,0.5), max.words=100, random.order=FALSE, rot.per=0.35, use.r.layout=FALSE, colors=brewer.pal(8, 'Dark2'))

#Finding high frequency phrases or 'ngrams'

myTDM <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))#, weighting = weightTfIdf))

rowTotals <-  row_sums(myTDM)

write.csv(rowTotals, "Unigram.csv")

#find correlations

#findAssocs(myTDM, 'variety', 0.05)

##################################
####  BLOCK 2 Wordcloud Ends   ###
##################################



##################################
####  BLOCK 3 nGramming       ###
##################################


#Looking for phrases

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

BigramTDM <- TermDocumentMatrix(myCorpus, control = list(tokenize = BigramTokenizer))

rowTotals <-  row_sums(BigramTDM)

write.csv(rowTotals, "Bigram.csv")

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))

TrigramTDM <- TermDocumentMatrix(myCorpus, control = list(tokenize = TrigramTokenizer))

rowTotals <-  row_sums(TrigramTDM)

write.csv(rowTotals, "Trigram.csv")

#QuadgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))

#QuadgramTDM <- TermDocumentMatrix(myCorpus, control = list(tokenize = QuadgramTokenizer))

#rowTotalsQ <-  row_sums(QuadgramTDM)

#write.csv(rowTotalsQ, "Quadgram.csv")

#megaTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 4))

#megaTDM <- TermDocumentMatrix(myCorpus, control = list(tokenize = megaTokenizer))

#rowTotals <-  row_sums(megaTDM)

#write.csv(rowTotals, "mega.csv")

##################################
####  BLOCK 3 nGramming ends   ###
##################################

##################################
####  BLOCK 4 Sentiment        ###
##################################

SimplifyText <- function(x) {
  return(removePunctuation(removeNumbers(stripWhitespace(tolower(x))))) 
}


fulltext = as.character(MH_Extract$Comments)

#posboost = as.character(posboost$V1)

#negboost = as.character(negboost$V1)

#positive.words = c(positive.words, posboost)

#negative.words = c(negative.words, negboost)

boostSentiment <- sentiment_frame(positive.words, negative.words)

sentiment=polarity(SimplifyText(fulltext),polarity.frame=boostSentiment)$all$polarity

MH_Extract<-data.frame(MH_Extract,Sentiment_Polarity=sentiment)

write.csv(MH_Extract,"MH_Sentiments.csv") #, sep=",")

##################################
####  BLOCK 4 Sentiment ends   ###
##################################






#################################################################
### Text Mining for Fun and Profit (c)Saurabh Choudhary, 2014 ###
#################################################################