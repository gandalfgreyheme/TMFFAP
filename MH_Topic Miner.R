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

#k=30 #Number of topics for LDA

MH_Extract<-read.csv("AS_LDA_Corpus_TabOnly.csv",header=TRUE) #import the full CSV file of the data import

MH_Extract$Comments[MH_Extract$Comments==""] <- NA

MH_Extract <- MH_Extract[is.na(MH_Extract$Comments)==0,]

#MH_Extract <- MH_Extract[sample(1:nrow(MH_Extract), 10000, replace=FALSE),]

SimplifyText <- function(x) {
  return(removePunctuation(removeNumbers(stripWhitespace(tolower(x))))) 
}

fulltext = as.character(MH_Extract$Comments)
 

text<-fulltext

text<-stri_replace_all_regex(text,"[^[:alnum:]///' ]"," ")

text<-stri_replace_all_regex(text,"[^[:alnum:] ]","")

text<-removeWords(text,stopwords('english'))

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
                 "pls", "provide", "provided","providing","time", "visit", "visited",
                 "visiting","require","required","increase","increasing","increased",
                 "people","regular","expect","expected","expecting","add","added",
                 "found","bring","inform","informed","person","select","selected",
                 "call","calling","called","half","feedback","age","arrange",
                 "arrangement","bit","bought","care","carry","cheers","choice",
                 "choices","choose","chosen","company","day","diwali","exchange",
                 "explain","gift","gr","hope","items","larger","lot","luv",
                 "maintain","mall","outlet","pl","play","purchase","purchased",
                 "requirement","requirements","response","rest","send","special",
                 "specially","thnx")

text<-removeWords(text,myStopwords)

text<-removeWords(text,positive.words)

nw1=negative.words[1:1500] #splitting qdap dictionary because of regex limit of ~2k words

nw2=negative.words[1501:3000]

nw3=negative.words[3001:4776]

text<-removeWords(text,nw1)

text<-removeWords(text,nw2)

text<-removeWords(text,nw3)

text <- removeWords(text, negation.words)

text <- removeWords(text, BuckleySaltonSWL)

text <- removeWords(text, OnixTxtRetToolkitSWL1)

text <- removeWords(text, deamplification.words)

text <- removeWords(text, preposition)

text<-removePunctuation(removeNumbers(stripWhitespace(tolower(text))))

stemDocumentfix <- function(x)
{
  PlainTextDocument(paste(stemDocument(unlist(strsplit(as.character(x), " "))),collapse=' '))
}

trans<-read.csv("Translator.csv",header=TRUE)

w<-as.character(trans$Word)

t<-as.character(trans$Trans)

text<-stri_replace_all_fixed(text, w, t, vectorize_all = FALSE)

tpcs=matrix(nrow=length(text),ncol=1)

tpcs<-data.frame(tpcs)

topic_list<-read.csv("Topics_List.csv",header=FALSE)


for (i in 1:length(topic_list$V1))
{
  found=grep(as.character(topic_list[i,1]),text)
  for(j in 1:length(found))
  {
    if(is.na(tpcs[found[j],1]))
    {
      tpcs[found[j],1]=as.character(topic_list[i,1])
    }
    else
    {
      tpcs[found[j],1]=paste(tpcs[found[j],1],as.character(topic_list[i,1]))
    }
  }
}


#myCorpus <- tm_map(myCorpus, stemDocumentfix)   #stems words to word roots e.g. manager, managed, managing -> manag

#myCorpus <- tm_map(myCorpus, PlainTextDocument)

#myTDM <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))#, weighting = weightTfIdf))

#rowTotals <-  row_sums(myTDM)

#write.csv(rowTotals, "WordList.csv")


top=as.character(tpcs$tpcs)


original<-sentiment_frame(positive.words,negative.words,
                          pos.weights=1,neg.weights=-1)

neg<-sentiment_frame(positive.words,negative.words,
                     pos.weights=1,neg.weights=-10)

negative.words=c(negative.words,negation.words)

superneg<-sentiment_frame(positive.words,negative.words,
                          pos.weights=1,neg.weights=-10)

#pole2=polarity(SimplifyText(fulltext), 
#              polarity.frame = qdapDictionaries::key.pol, constrain = FALSE,
#             negators = qdapDictionaries::negation.words,
#              amplifiers = qdapDictionaries::amplification.words,
#              deamplifiers = qdapDictionaries::deamplification.words,
#              question.weight = 0, amplifier.weight = 0.8, n.before = 4,
#              n.after = 2, rm.incomplete = FALSE)


########################################################################
# The pole function has been modified below.
# If polarity.frame = superneg, then the algo will aggressively penalise for
# negative words
# If polarity.frame = neg, then algo will prioritise negative words
# If polarity.frame = original, then the algo gives equal importance to positive
# and negative words
# n.before and n.after set to 0 as we're stripping punctuation. Because of which
# modifiers start misbehaving e.g.:
# Original text = Poor. will never ever come back
# Stripped text = poor will never ever come back
# Therefore -> polar word = "poor", modified by "never" negative of negative = 
# positive - bad for bijniss
########################################################################

pole=polarity(SimplifyText(fulltext),polarity.frame=superneg,n.before=0,n.after=0
              , constrain=TRUE) 


sentiment=pole$all$polarity 

#npole=polarity(SimplifyText(fulltext),polarity.frame=neg,n.before=0,n.after=0
#                , constrain=TRUE)$all$polarity
#pole=polarity(SimplifyText(fulltext),polarity.frame=original,n.before=0,n.after=0
#                , constrain=TRUE)$all$polarity

MH_Extract<-data.frame(MH_Extract,Sentiment=sentiment,Topic=top)

write.csv(MH_Extract,file=paste("Sentiments.csv"))
