library(datadr)
library(dplyr)
library(ggplot2)
library(glmnet)

library(extraDistr)

load("exercise1.RData")

##### ----------------------- Define X, y, n, p, m, s, r ----------------------- 
X <- as.data.frame(XB1) %>% as.matrix %>% scale
y <- as.data.frame(yB1) %>% as.matrix %>% as.numeric()
colnames(y) <-  "y"

n <- dim(X)[1] # Total num of sample points 
p <- dim(X)[2] # num of features

data_all <- cbind(y, X) %>% as.data.frame()



##### ----------------------- Fit a linear model on the scaled all features ----------------------- 
fit.all <- lm(y~., data=data_all)








#####################################################################################
##### ----------------------- Methods of feature selection ----------------------- 
#####################################################################################
##### ----------------------- Feature selection by lasso ----------------------- 
Lasso_cv <- cv.glmnet(X, y, alpha = 1,
                      type.measure = "mse",
                      ## K = 10 is the default.
                      nfold = 10,## prevalidated array is returned
                      #lambda = c(0.01, 0.5),
                      keep = TRUE)

Lasso_cv$lambda.1se
Coef_lasso <- coef(Lasso_cv, s=Lasso_cv$lambda.1se)[-1] %>% as.numeric()
# Coef_lasso <- coef(Lasso_cv)[-1] %>% as.numeric()

(features_by_lasso <- which(Coef_lasso!=0))


# the selected features by lasso
X_selected <- X[,features_by_lasso]
data_selected <- cbind(y, X[,features_by_lasso]) %>% as.data.frame()


##### ----------------------- Fit a linear model from data_selected by lasso  ----------------------- 
fit.selected <- lm(y~., data=data_selected)




##### ----------------------- Stepwise feature selection, using AIC ----------------------- 
##### ----------------------- Super slow in B1 ----------------------- 

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
model <- randomForest(X[,features_by_lasso], y, ntree=500, importance=T, 
                      keep.forest=T, replace = F, 
                      sampsize = 0.8*nrow(X))
plot(model)
varImpPlot(model, type=1,  main = "% of increasing in MSE, Dataset B1[, selected_by_Lasso]") # Optional arg: n.var = 100,

imp_table <- NULL
for (i in (1:20)){
  model <- randomForest(X[,features_by_lasso], y, ntree=100, importance=T, 
                        keep.forest=T, replace = F, 
                        sampsize = 0.8*nrow(X))
  imp_table <- cbind(imp_table, model$importance[,1])
}

avg_imp_rf <- rowMeans(imp_table) %>% as.data.frame()

# Compute the average VI by random forest
o <- order(avg_imp_rf,decreasing = T)
avg_imp_rf_ranking <- avg_imp_rf[order(avg_imp_rf, decreasing = T),] %>% as.data.frame()
plot(avg_imp_rf)

# Save the VI to a variable
rfImp <- model$importance
rfImp_sort <- rfImp[order(rfImp[,1], decreasing = T), ] # The decreasing order in %IncMSE

features_by_rf <- (order(rfImp[,1], decreasing = T)[1:(2*length(features_by_lasso))]) %>% sort # The first few important by IncMSE




## Find the common features selected by Random forest and Lasso
# (features_shared_LS_rf <- features_by_rf[features_by_rf %in% features_by_lasso])




##### -----------------------  Calculate Relative Importance, by "lmg" ----------------------- 
# -------- Due to so many features, lmg is super slow on ALL features.
#          Then I tried LMG with SELECTED features by elastic net, unfortunately, too slow!
library(relaimpo)
metrics <- calc.relimp(fit.selected,type=c("lmg"), rela=T)
metrics
lmg_imp <- metrics$lmg %>% as.data.frame()




#####################################################################################
##### -----------------------  Fit the final model, and compute MSE ##### -----------------------  
#####################################################################################
print("------------Coefficients by lm()-------------")
(coef.selectedVar <- coef(fit.selected))




