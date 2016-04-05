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

MH_Extract<-read.csv("AS_Feedback_Corpus.csv",header=TRUE) #import the full CSV file of the data import

MH_Extract$Comments[MH_Extract$Comments==""] <- NA

MH_Extract <- MH_Extract[is.na(MH_Extract$Comments)==0,]

MH_Extract <- MH_Extract[sample(1:nrow(MH_Extract), 3000, replace=FALSE),]

rawVerbatim <- data.frame(MH_Extract$Comments)

myCorpus <- Corpus(DataframeSource(rawVerbatim))

#myTDM <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))#, weighting = weightTfIdf))

#rowTotals <-  row_sums(myTDM)

#write.csv(rowTotals, "WordList.csv")

toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x))})

myCorpus <- tm_map(myCorpus, toSpace, "-")

myCorpus <- tm_map(myCorpus, toSpace, "'")

#myCorpus <- tm_map(myCorpus, toSpace, "'")

#myCorpus <- tm_map(myCorpus, toSpace, ".")

#myCorpus <- tm_map(myCorpus, toSpace, ".")

#myCorpus <- tm_map(myCorpus, toSpace, """)

myCorpus <- tm_map(myCorpus, toSpace, ",")

myCorpus <- tm_map(myCorpus, removePunctuation)   #removes punctuation

myCorpus <- tm_map(myCorpus, removeNumbers)  #removes all numbers

myCorpus <- tm_map(myCorpus, stripWhitespace)  #strip extra spaces

myCorpus <- tm_map(myCorpus, tolower)  #converts all test to lower case

myCorpus <- tm_map(myCorpus, removeWords, stopwords('english'))  #removes all stop words





#Remove custom Stop Words

myStopwords <- c("the","i","and","was","in","of","with","to","am","were","for",
                 "at","is",".","have","a","they","it","me","all","what",
                 "that","from","there","you","my","so","be","should","also","as",
                 "are","had","we","did","which","customer","get","your","this",
                 "any","much","overall","on","would","about","when","able",
                 "do","t","if","some","has","he","can","will","us","well","&",
                 "its","or","by","given",",i","could","gave","because","m",
                 "feedback","-","for.","an","give","after", "said")

myCorpus <- tm_map(myCorpus, removeWords, myStopwords)


myCorpus <- tm_map(myCorpus, stemDocument)   #stems words to word roots e.g. manager, managed, managing -> manag

myCorpus <- tm_map(myCorpus, PlainTextDocument)

dtm.temp<-DocumentTermMatrix(myCorpus)

dtm.temp$dimnames$Docs<-as.character(MH_Extract$UID)

RT <- apply(dtm.temp , 1, sum)

dtm<-dtm.temp[RT>0,]

library(topicmodels)

#Set parameters for Gibbs sampling

burnin <- 4000

iter <- 2000

thin <- 500

seed <-list(2003,5,63,100001,765)

nstart <- 5

best <- TRUE





#Number of topics

k <- 30




#Run LDA using Gibbs sampling

ldaOut <-LDA(dtm,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

#write out results

#docs to topics

ldaOut.topics <- as.matrix(topics(ldaOut))

write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

#top 6 terms in each topic

ldaOut.terms <- as.matrix(terms(ldaOut,10))

write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))

#probabilities associated with each topic assignment

topicProbabilities <- as.data.frame(ldaOut@gamma)

write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

#Find relative importance of top 2 topics

topic1ToTopic2 <- lapply(1:nrow(dtm),function(x)
  
  sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])

#Find relative importance of second and third most important topics

topic2ToTopic3 <- lapply(1:nrow(dtm),function(x)
  
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])

#write to file


#write to file

write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))

write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))