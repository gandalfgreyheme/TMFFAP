CVP<-read.csv(file.choose(),header=TRUE)
str(CVP)
summary(CVP)
CVP_Analyze<-CVP[,3:12]
cor(CVP_Analyze)
pairs(CVP_Analyze)
#summary(allin)
allin = lm(Desirability ~ VFM + Cols_Choice + Fash_Fit 
           + Innov_Style + Rightness_Fit + Fash_Styles 
           + Qual_Store, data=CVP_Analyze)
step(allin, direction="backward")
desire_step = lm(Desirability ~ Innov_Style + Rightness_Fit 
                 + Qual_Store, data=CVP_Analyze)