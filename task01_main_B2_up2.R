library(datadr)
library(dplyr)
library(ggplot2)
library(glmnet)

library(extraDistr)
library(foreach)

load("exercise1.RData")

##### ----------------------- Define X, y, n, p, m, s, r ----------------------- 
X <- as.data.frame(XB2) %>% as.matrix %>% scale
y <- as.data.frame(yB2) %>% as.matrix %>% as.numeric()
colnames(y) <-  "y"

n <- dim(X)[1] # Total num of sample points 
p <- dim(X)[2] # num of features

data_all <- cbind(y, X) %>% as.data.frame()




##### ----------------------- Fit a linear model on the scaled all features ----------------------- 

## --------- fit a linear model use lm --------- 
fit.all <- lm(y~., data=data_all)




# ---------------------- Feature selection by a subsample of B2, using Lasso ----------------
sub_indice <- sample.int(n, 10000, replace = F)

Lasso_cv.subInd <- cv.glmnet(X[sub_indice, ], y[sub_indice], alpha = 1,
                      type.measure = "mse",
                      ## K = 10 is the default.
                      nfold = 10,## prevalidated array is returned
                      keep = TRUE)
Lasso_cv.subInd$lambda.1se

Coef_lasso.subInd <- coef(Lasso_cv.subInd, s=Lasso_cv.subInd$lambda.1se)[-1] %>% as.numeric()

(features_by_lasso.subInd <- which(Coef_lasso.subInd!=0))






#####################################################################################
# ----------------------- Methods of feature selection ----------------------- 
#  ####################################################################################
##### ----------------------- Feature selection by lasso ----------------------- 
Lasso_cv <- cv.glmnet(X, y, alpha = 1,
                      type.measure = "mse",
                      ## K = 10 is the default.
                      nfold = 10,## prevalidated array is returned
                      keep = TRUE)
Lasso_cv$lambda.1se

Coef_lasso <- coef(Lasso_cv, s=Lasso_cv$lambda.1se)[-1] %>% as.numeric()

(features_by_lasso <- which(Coef_lasso!=0))


# the selected features by lasso
X_selected <- X[,features_by_lasso]
data_selected <- cbind(y, X[,features_by_lasso]) %>% as.data.frame()



##### ----------------------- Fit a linear model from data_selected by lasso  ----------------------- 
system.time({
fit.selected <- lm(y~., data=data_selected)
})



##### ----------------------- Use package vita ----------------------- 
# library(vita)
# library("randomForest")
# reg.rf= randomForest(X_selected,y,mtry = 3,ntree=100,
#                      importance=TRUE,keep.inbag = TRUE)


##### ----------------------- Stepwise feature selection, using AIC ----------------------- 
library(caret)

train.control <- trainControl(method = "cv", number = 20)

# http://topepo.github.io/caret/train-models-by-tag.html#generalized-linear-model
regfit.fwd <-  caret::train(X_selected, as.numeric(y), method = "lmStepAIC", # "leapForward", "leapSeq"
                            trControl = train.control, intercept = FALSE)
summary(regfit.fwd)




##### ----------------------- Random forest variable importance ----------------------- 
# library( randomForest )
# model <- randomForest(X, y, importance=T)
# varImpPlot(model)

library( randomForest )

library( iterators )
library(foreach)
library(doParallel)
registerDoParallel(numCores)  # use multicore, set to the number of our cores


### --------- parallel computing -------------
num_cores <- detectCores(all.tests = FALSE, logical = TRUE)

cl <- makeCluster(num_cores)

registerDoParallel(cl)

system.time({
  para.data <- foreach (i=1:200, .combine=cbind, .packages='randomForest') %dopar% {
    rand.indice <- sample.int(n, 10000, replace = F)
    X.rfsub <- X[rand.indice,features_by_lasso]
    y.rfsub <- y[rand.indice]
    model <- randomForest(X.rfsub, # 
                          y.rfsub, ntree=7, importance=T, 
                          keep.forest=F, replace = T, 
                          sampsize = 0.5*nrow(X.rfsub))
    model$importance[,1]
  }
  avg_imp_rf <- rowMeans(para.data) %>% as.data.frame()
})

stopCluster(cl)



### --------- A single random forest ---------------
model <- randomForest(X[,features_by_lasso], y, ntree=80, importance=T, 
                      keep.forest=T, replace = F, 
                      sampsize = 0.05*nrow(X))
plot(model)
varImpPlot(model, type=1,  main = "% of increasing in MSE, Dataset B1[, selected_by_Lasso]") # Optional arg: n.var = 100,


### ----------- For loop ----------------


system.time({
  imp_table <- NULL
  for (i in (1:2)){
    model <- randomForest(X[,features_by_lasso], y, ntree=80, importance=T, 
                          keep.forest=T, replace = F, 
                          sampsize = 0.05*nrow(X))
    imp_table <- cbind(imp_table, model$importance[,1])
  }
})

avg_imp_rf <- rowMeans(imp_table) %>% as.data.frame()

# Compute the average VI by random forest
o <- order(avg_imp_rf,decreasing = T)
avg_imp_rf_ranking <- avg_imp_rf[order(avg_imp_rf, decreasing = T),] %>% as.data.frame()



##### -----------------------  Calculate Relative Importance, by "lmg" ----------------------- 
# -------- Due to so many features, lmg is super slow on ALL features.
#          Then I tried LMG with SELECTED features by lasso.
library(relaimpo)
metrics <- calc.relimp(fit.selected,type=c("lmg"), rela=T)
lmg_imp <- metrics$lmg %>% as.data.frame()



##### -----------------------  Final model ----------------------- 
(coef.final <- coef(fit.selected))


# --------- fit a linear model use bag of little bootstrap --------- 

rrkc <- datadr::divide(
  data_selected, by = datadr::rrDiv(500), update = TRUE)

system.time({
kcBLB <- rrkc %>% datadr::addTransform(function(x) {
  drBLB(
    x,
    statistic = function(x, weights) {
      coef(glm(y ~ ., data = x, family = "gaussian", weights = weights))
    },
    metric = function(x) {
      quantile(x, c(0.025, 0.975))
    },
    R = 100,
    n = nrow(rrkc))
})

coefs <- datadr::recombine(kcBLB, datadr::combMean)
matrix(coefs, ncol = 2, byrow = TRUE)
})





