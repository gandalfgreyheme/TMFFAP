# Market Basket Analysis - Adapted from 
# http://www.salemmarafi.com/code/market-basket-analysis-with-r/

# Load the libraries
library(arules)
library(arulesViz)
library(datasets)

# set working directory
setwd("D:/R_Out")

# Create data set from bill evel info
# NOTE: The bill data must be stored as a text file with each line containing two
# items - the bill no and the item. The data should be tab delimited
VHBill <- read.transactions("VH_CP_Flat1.txt", format = "single", cols = c(1,2), rm.duplicates=TRUE)

# Create an item frequency plot for the top 20 items
itemFrequencyPlot(VHBill,topN=20,type="absolute")


# Set support and confidence values
support=0.002
confidence=0.7

# Get the rules
# rules <- apriori(VHBill, parameter = list(supp = 0.001, conf = 0.8))

# Show the top 5 rules, but only 2 digits
# options(digits=2)
# inspect(rules[1:5])

#Sorting rules by confidence
# rules<-sort(rules, by="confidence", decreasing=TRUE)

#Lets say you wanted more concise rules. That is also easy to do by adding a 
#"maxlen" parameter to your apriori function
rules <- apriori(VHBill, parameter = list(supp = support, conf = confidence,minlen=1,maxlen=3))
#rules <- apriori(VHBill, parameter = list(supp = support, conf = confidence,maxlen=2))
#remove redundant rules generated
subset.matrix <- is.subset(rules, rules)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
rules.pruned <- rules[!redundant]
rules<-rules.pruned

# What are customers likely to buy before buying ITEM_X?
# What are customers likely to buy if they purchase ITEM_X?

rules_p<-apriori(data=VHBill, parameter=list(supp=support,conf = confidence, minlen=1,maxlen=3), 
               appearance = list(default="lhs",rhs="VDFGJEANS"),
               control = list(verbose=F))
rules_p<-sort(rules_p, decreasing=TRUE,by="confidence")
 inspect(rules_p[1:5])

# we can set the left hand side to be "ITEM_X" and find its antecedents

rules_a<-apriori(data=VHBill, parameter=list(supp=support,conf = confidence,minlen=1,maxlen=3), 
               appearance = list(default="rhs",lhs="VDFGJEANS"),
               control = list(verbose=F))
rules_a<-sort(rules_a, decreasing=TRUE,by="confidence")
# inspect(rules_a[1:5])

# Visualize Rules
plot(rules_p,method="graph",interactive=TRUE)

write(rules.pruned, file = "AllRules.csv", quote=TRUE, sep = ",", col.names = NA)
write(rules_a, file = "Rules_A.csv", quote=TRUE, sep = ",", col.names = NA)
write(rules_p, file = "Rules_P.csv", quote=TRUE, sep = ",", col.names = NA)