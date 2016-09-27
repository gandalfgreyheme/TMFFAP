doInstall <- FALSE  # Change to FALSE if you don't want packages installed.
toInstall <- c("jpeg", "reshape2", "ggplot2")
if(doInstall){install.packages(toInstall, repos = "http://cran.r-project.org")}
lapply(toInstall, library, character.only = TRUE)

library(jpeg)
library(ggplot2)
library(reshape2)

imageLoader <- function(url){  # This function takes a URL, and generates a data.frame with pixel locations and colors
  # Download to disk, load
  #download.file(url, "tempPicture.jpg", mode = "wb")  # Stash image locally
  readImage <- readJPEG(url)
  #readImage <- readJPEG("C:\Analytics\RGB\63903-158213-small.jpg")
  
  longImage <- melt(readImage)
  rgbImage <- reshape(longImage, timevar = "Var3",
                      idvar = c("Var1", "Var2"), direction = "wide")
  rgbImage$Var1 <- -rgbImage$Var1
  return(rgbImage)
}

##########
# Part 2 # Identifying "dominant" colors with k-means
##########

loc<-"C:/Analytics/RGB/63903-158213-small.jpg"
rgbImage <- imageLoader(loc)  # Pick one, or use your own URL.
with(rgbImage, plot(Var2, Var1, col = rgb(rgbImage[, 3:5]), asp = 1, pch = "."))

# Cluster in color space:
kColors <- 5  # Number of palette colors
kMeans <- kmeans(rgbImage[, 3:5], centers = kColors)

zp1 <- qplot(factor(kMeans$cluster), geom = "bar",
             fill = factor(kMeans$cluster))
zp1 <- zp1 + scale_fill_manual(values = rgb(kMeans$centers))
zp1

approximateColor <- kMeans$centers[kMeans$cluster, ]
qplot(data = rgbImage, x = Var2, y = Var1, fill = rgb(approximateColor), geom = "tile") +
  coord_equal() + scale_fill_identity(guide = "none")

colvect<-as.vector(kMeans$centers)
