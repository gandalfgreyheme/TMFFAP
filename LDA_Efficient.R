#library(RWeka)

library(tm)
library(stringi)
#library(SnowballC)

#library(wordcloud)

library(slam)

library(qdap)

library(plyr)

library(stringr)

#Import verbatim data - copy the verbatim and paste in a single row of a CSV file, do a remove duplicates function in excel before saving

setwd("D:/R_Out")

#memory.limit(size=memory.limit()+20)

k=10 #Number of topics for LDA

MH_Extract<-read.csv("AS_LDA_Corpus_TabOnly.csv",header=TRUE) #import the full CSV file of the data import

MH_Extract$Comments[MH_Extract$Comments==""] <- NA

MH_Extract <- MH_Extract[is.na(MH_Extract$Comments)==0,]

MH_Extract <- MH_Extract[sample(1:nrow(MH_Extract), 10000, replace=FALSE),]

SimplifyText <- function(x) {
  return(removePunctuation(removeNumbers(stripWhitespace(tolower(x))))) 
}

fulltext = as.character(MH_Extract$Comments)

#pole=polarity(SimplifyText(fulltext),)

pole2=polarity(SimplifyText(fulltext), 
              polarity.frame = qdapDictionaries::key.pol, constrain = FALSE,
              negators = qdapDictionaries::negation.words,
              amplifiers = qdapDictionaries::amplification.words,
              deamplifiers = qdapDictionaries::deamplification.words,
              question.weight = 0, amplifier.weight = 0.8, n.before = 4,
              n.after = 2, rm.incomplete = FALSE)

sentiment=pole$all$polarity

MH_Extract<-data.frame(MH_Extract,Sentiment_Polarity=sentiment)

write.csv(MH_Extract,file=paste("LDA",k,"Sentiments.csv")) 

rawVerbatim <- data.frame(MH_Extract$Comments)

myCorpus <- Corpus(DataframeSource(rawVerbatim))

#myTDM <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))#, weighting = weightTfIdf))

#rowTotals <-  row_sums(myTDM)

#write.csv(rowTotals, "WordList.csv")

onlyAlpha <- content_transformer(function(x) stri_replace_all_regex(x,"[^[:alnum:]///' ]"," "))

myCorpus <- tm_map(myCorpus, onlyAlpha)

stripQuotes <- content_transformer(function(x) stri_replace_all_regex(x,"[^[:alnum:] ]",""))

myCorpus <- tm_map(myCorpus, stripQuotes)

#toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x))})

#myCorpus <- tm_map(myCorpus, toSpace, "-")

#myCorpus <- tm_map(myCorpus, toSpace, "'")

#myCorpus <- tm_map(myCorpus, toSpace, "'")

#myCorpus <- tm_map(myCorpus, toSpace, ".")

#myCorpus <- tm_map(myCorpus, toSpace, ".")

#myCorpus <- tm_map(myCorpus, toSpace, """)

#myCorpus <- tm_map(myCorpus, toSpace, ",")

#myCorpus <- tm_map(myCorpus, removePunctuation)   #removes punctuation

myCorpus <- tm_map(myCorpus, removeNumbers)  #removes all numbers

myCorpus <- tm_map(myCorpus, tolower)  #converts all test to lower case

myCorpus <- tm_map(myCorpus, removeWords, stopwords('english'))  #removes all stop words





#Remove custom Stop Words

myStopwords <- c("a","about","after","all","allen","allensolly","also","am",
                 "an","and","any","are","as","at","b","back","be","because",
                 "been","but","buy","by","came","can","comment","comment.",
                 "comments","comments.","could","customer","customers","d",
                 "da","did","didnt","dint","do","does","doesnt","done","even",
                 "ever","every","everything","fine","for","from","get","gets",
                 "give","given","good","goood","got","gr8","great","gud","guys",
                 "had","happy","has","have","he","here","i","if","ill","im","in",
                 "is","it","its","job","just","like","liked","lyk","m","me",
                 "mr","my","n","na","nd","nice","none","of","ok","on","or",
                 "our","overall","put","r","s","say","see","service","should",
                 "so","solly","some","stuff","t","than","thank","thanks","thanx",
                 "that","the","their","there","they","thing","this","to","too",
                 "u","up","upp","ur","us","very","wanna","want","wanted","wants",
                 "was","wat","we","wear","wears","well","were","what","when",
                 "which","will","wit","with","would","yes","you","your","yrs",
                 "plz","feel","really","help","please", "make","try", "one", "much", 
                 "pls")

myCorpus <- tm_map(myCorpus, removeWords, myStopwords)

myCorpus <- tm_map(myCorpus, stripWhitespace)  #strip extra spaces

stemDocumentfix <- function(x)
{
  PlainTextDocument(paste(stemDocument(unlist(strsplit(as.character(x), " "))),collapse=' '))
}

myCorpus <- tm_map(myCorpus, stemDocumentfix)   #stems words to word roots e.g. manager, managed, managing -> manag

myCorpus <- tm_map(myCorpus, PlainTextDocument)

dtm.temp<-DocumentTermMatrix(myCorpus)

dtm.temp$dimnames$Docs<-as.character(MH_Extract$UID)


rm(MH_Extract)
rm(myCorpus)
rm(myStopwords)
rm(rawVerbatim)

gc()

RT <- apply(dtm.temp , 1, sum)

dtm<-dtm.temp[RT>3,] #Include comments more than 3 words

rm(dtm.temp)
rm(RT)

gc()

library(topicmodels)

#Set parameters for Gibbs sampling

burnin <- 4000

iter <- 2000

thin <- 500

seed <-list(2003,5,63,100001,765)

nstart <- 5

best <- TRUE



#Run LDA using Gibbs sampling

ldaOut <-LDA(dtm,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

#write out results

#docs to topics

ldaOut.topics_3 <- as.matrix(topics(ldaOut,3))

write.csv(ldaOut.topics_3,file=paste("LDAGibbs",k,"DocsToTopics_3.csv"))

#top 6 terms in each topic

ldaOut.terms <- as.matrix(terms(ldaOut,6))

write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))



#probabilities associated with each topic assignment

# topicProbabilities <- as.data.frame(ldaOut@gamma)

# write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

#Find relative importance of top 2 topics

# topic1ToTopic2 <- lapply(1:nrow(dtm),function(x)
  
  #sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])

#Find relative importance of second and third most important topics

#topic2ToTopic3 <- lapply(1:nrow(dtm),function(x)
  
 # sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])

#write to file


#write to file

#write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))

#write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))