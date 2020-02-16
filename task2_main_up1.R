library(tidyverse)
library(grid)
library(latex2exp) # Latex in ggplot2 labels

library(subspace)   # Kind of an annoying package. Hard to install and has to
# be loaded globally. However, this seems to be the only
# reasonable package for subspace clustering in R
library(ggraph)     # Used for plotting of graphs
library(data.table) # Used for reading data from an URL, but this is actually
# a very good implementation of data frames that is highly
# optimized for large scale computation on a single machine

library(glmnet)
library(Rtsne) # for tSNE
library(mclust)  # Used for GMM clustering


my_theme <- theme_bw() + 
            theme(plot.title = element_text(hjust = 0.5, 
                                            size = 14, 
                                            face="bold"), 
            plot.subtitle = element_text(hjust = 0.5)) + theme(axis.title=element_text(size=12))


cbPalette <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # colour-blind friendly palette







## ------------------------ Load data ------------------------
load('exercise2.RData')
data <- as_tibble(X)
data_scaled <- scale(data)

p <- ncol(data)





## ------------------------ findCorrelation: Determine highly correlated variables ------------------------
library(caret)
findCorrelation(data_scaled, cutoff = 0.9, verbose = FALSE, names = FALSE,
                exact = F)

library(corrr)
corrVar <- data_scaled %>%
  correlate() %>% 
  stretch() %>% 
  arrange(r)


corrVar.beta <- var(data_scaled)





## ------------------------ dip test of clustering ------------------------
library(diptest)
dip_test <- NULL
for (i in 1:p){
  dip_test <- c(dip_test,
                dip.test(data_scaled[,i], simulate.p.value = F, B = 2000)$p.value)
}

(important_features <- which(dip_test < 0.01))
#(important_features <- order(dip_test, decreasing = F)[1:20] %>% sort() )
# c(321, 411, 519, 547) 
data_important <- data_scaled[,important_features]

## An important feature
hist(data_scaled[,321], main="Histogram of an important feature", xlab = NULL)
dip_test[321]

## An unimportant feature
hist(data_scaled[,1], main="Histogram of an unimportant feature", xlab = NULL)
dip_test[1]




## ------------------------ GMM clustering + silhouette-width  ------------------------------
library(cluster)

## Define a function that returns the plot and value of average silhouette width
GMM_silhouette <- function(num_clusters, data){
  
  clust_G <- Mclust(data = data, G=num_clusters)
  
  s <- cluster::silhouette(clust_G$classification, daisy(data, metric = "euclidean"))
  
  # average width
  avg_width <- mean(s[,3])
  
  data_plot <- tibble(
    obs = as.factor(seq(1,dim(data)[1])),
    cluster = as.factor(s[,1]),
    sw = s[,3]) %>%   # Silhouette width
    arrange(cluster, sw) %>%
    mutate(obs = factor(obs, levels = obs))
  
  cbPalette <- c(
    "#999999", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # colour-blind friendly palette
  
  plt <- ggplot(data_plot) +
    geom_bar(aes(x = obs, y = sw, fill = cluster), stat = "identity") +
    coord_flip() +
    scale_x_discrete("Observation") +
    scale_y_continuous(
      "Silhouette Width", breaks = seq(-1, 1, by = 0.2)) +
    scale_fill_manual(values = cbPalette[-1], guide = FALSE) +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text.y = element_blank(),
      axis.text.x = element_text(size = 7),
      axis.title = element_text(size = 7))
  
  return(list(num_of_clusters=num_clusters, avg_SilhouetteWidth=avg_width, SilhouetteWidth_plot=plt))
  
}



## Plot the average silhouette width vs number of clusters
cluster_num_range <- 2:10
avg_silWidth <- sapply(cluster_num_range, function (x) {GMM_silhouette(x, data_important)$avg_SilhouetteWidth})

PlotData_avg_silWidth <- tibble(cluster_num_range, avg_silWidth)

ggplot(PlotData_avg_silWidth) + 
  aes(x=cluster_num_range, y=avg_silWidth) + geom_line() + geom_point() +
  # geom_hline(yintercept=0.8, color='red', size = 0.1) + 
  # annotate("text", max(cluster_num_range), 0.8, vjust = -1, label = "Cutoff")+
  #ggtitle('Average Silhouette Width') +
  labs(title = 'Average Silhouette Width (17 features selected by dip test)') +
  my_theme + 
  xlab('Number of clusters') + ylab('Average Silhouette Width')+
  scale_x_continuous(breaks = seq(0,10,1))



## Plot SilhouetteWidth_plot of cluster number = 3 (the optimal, largest average solwidth)
GMM_silhouette(3, data_important)$SilhouetteWidth_plot



## ------------------------GMM on important features by dip test ------------------------
num_clusters <- 3
GMM_clust <- Mclust(data_important, num_clusters)

write.csv(GMM_clust$classification, file = "cluster_assignment_17features.csv")


## ---- tSNE on important data----
perplexity_ <- 30
tSNE_result <- Rtsne(data_important, 
                     dims = 2,
                     perplexity = perplexity_,
                     num_threads = 1,
                     pca = T,
                     pca_center = T, pca_scale = F)

data_tSNE_toplot <- tibble(tSNE1 = tSNE_result$Y[, 1],
                           tSNE2 = tSNE_result$Y[, 2])

ggplot(data_tSNE_toplot, aes(tSNE1, tSNE2, color=as.factor(GMM_clust$classification))) +
  geom_point() +
  scale_colour_manual(values=cbPalette) +  my_theme + theme(legend.position = 'none')
   #labs(title = "tSNE of the simulated data", subtitle = sprintf('Perplexity = %d',perplexity_)) +
  





# ------------------------Randomly selected some features, and plot the silhouettewidth-----
if (0){
  
  cluster_num_range <- 2:10
  avg_silWidth.rand <- NULL
  for (i in (1:10)){
    features_rand <- sample.int(n = p, size = length(important_features))
    avg_silWidth.rand.this <- sapply(cluster_num_range, function (x) {GMM_silhouette(x, data_scaled[,features_rand])$avg_SilhouetteWidth})
    avg_silWidth.rand <- cbind(avg_silWidth.rand, avg_silWidth.rand.this)
  }
  
  avg_silWidth.rand.avg <- rowMeans(avg_silWidth.rand)
  
  PlotData_avg_silWidth.rand <- tibble(cluster_num_range, avg_silWidth.rand.avg)
  
  ggplot(PlotData_avg_silWidth.rand) + 
    aes(x=cluster_num_range, y=avg_silWidth.rand.avg) + geom_line() + geom_point() +
    # geom_hline(yintercept=0.8, color='red', size = 0.1) + 
    # annotate("text", max(cluster_num_range), 0.8, vjust = -1, label = "Cutoff")+
    #ggtitle('Average Silhouette Width') +
    labs(title = 'Average Silhouette Width (randomly 17 features)') +
    my_theme + 
    xlab('Number of clusters') + ylab('Average Silhouette Width')+
  scale_x_continuous(breaks = seq(0,10,1))

}




## ------------------------ clustvarsel package, selection again ------------------------
library(clustvarsel)
library(doParallel)
out <- clustvarsel(data_important, G=3, direction = 'forward', parallel = T)
features_secondSelection <- out$subset

# Only five features in this
data_most_important <- data_important[,features_secondSelection]



# ------------------------Randomly 5 features from the 17 features in 1st round, and plot the silhouettewidth-----
if (0){
  
  cluster_num_range <- 2:10
  avg_silWidth.rand <- NULL
  for (i in (1:10)){
    features_rand <- sample.int(n = length(important_features), size = length(features_secondSelection))
    avg_silWidth.rand.this <- sapply(cluster_num_range, function (x) {GMM_silhouette(x, data_important[,features_rand])$avg_SilhouetteWidth})
    avg_silWidth.rand <- cbind(avg_silWidth.rand, avg_silWidth.rand.this)
  }
  
  avg_silWidth.rand.avg <- rowMeans(avg_silWidth.rand)
  
  PlotData_avg_silWidth.rand <- tibble(cluster_num_range, avg_silWidth.rand.avg)
  
  ggplot(PlotData_avg_silWidth.rand) + 
    aes(x=cluster_num_range, y=avg_silWidth.rand.avg) + geom_line() + geom_point() +
    # geom_hline(yintercept=0.8, color='red', size = 0.1) + 
    # annotate("text", max(cluster_num_range), 0.8, vjust = -1, label = "Cutoff")+
    #ggtitle('Average Silhouette Width') +
    labs(title = 'Average Silhouette Width (randomly 5 features from the 17 features)') +
    my_theme + 
    xlab('Number of clusters') + ylab('Average Silhouette Width')+
    scale_x_continuous(breaks = seq(0,10,1))
  
}





## ------------------------GMM on the 5 important features selected by clustval package ------------------------
num_clusters <- 3
GMM_clust_most <- Mclust(data_most_important, num_clusters)

write.csv(GMM_clust$classification, file = "cluster_assignment_clustvarsel.csv")


## ---- tSNE on the Most important 5 features ----
perplexity_ <- 30
tSNE_result <- Rtsne(data_important, 
                     dims = 2,
                     perplexity = perplexity_,
                     num_threads = 1,
                     pca = T,
                     pca_center = T, pca_scale = F)

data_tSNE_toplot <- tibble(tSNE1 = tSNE_result$Y[, 1],
                           tSNE2 = tSNE_result$Y[, 2])

ggplot(data_tSNE_toplot, aes(tSNE1, tSNE2, color=as.factor(GMM_clust_most$classification))) +
  geom_point() +
  scale_colour_manual(values=cbPalette) +
  #labs(title = "tSNE of the simulated data", subtitle = sprintf('Perplexity = %d',perplexity_)) +
  my_theme + theme(legend.position = 'none')



## Plot the average silhouette width vs number of clusters
cluster_num_range <- 2:10
avg_silWidth.most <- sapply(cluster_num_range, function (x) {GMM_silhouette(x, data_most_important)$avg_SilhouetteWidth})

PlotData_avg_silWidth.most<- tibble(cluster_num_range, avg_silWidth.most)

ggplot(PlotData_avg_silWidth.most) + 
  aes(x=cluster_num_range, y=avg_silWidth.most) + geom_line() + geom_point() +
  # geom_hline(yintercept=0.8, color='red', size = 0.1) + 
  # annotate("text", max(cluster_num_range), 0.8, vjust = -1, label = "Cutoff")+
  #ggtitle('Average Silhouette Width') +
  # labs(title = 'Average Silhouette Width (5 most important features)') +
  my_theme + 
  xlab('Number of clusters') + ylab('Average Silhouette Width')+
  scale_x_continuous(breaks = seq(0,10,1))


# Plot average silwidth of clustering based on the FIVE mose important features
GMM_silhouette(3, data_most_important)$SilhouetteWidth_plot
