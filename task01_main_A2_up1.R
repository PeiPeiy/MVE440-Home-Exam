library(datadr)
library(dplyr)
library(ggplot2)
library(glmnet)

library(extraDistr)

load("exercise1.RData")

##### ----------------------- Define X, y, n, p, m, s, r ----------------------- 
X <- as.data.frame(XA2) %>% as.matrix %>% scale
y <- as.data.frame(yA2) %>% as.matrix %>% as.numeric()
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
Lasso_cv <- cv.glmnet(X, y, alpha = 0.5,
                      type.measure = "mse",
                      ## K = 10 is the default.
                      nfold = 10,## prevalidated array is returned
                      keep = TRUE)

(Coef_lasso <- coef(Lasso_cv, s=Lasso_cv$lambda.1se)[-1] %>% as.numeric())

(features_by_lasso <- which(Coef_lasso!=0))


# the selected features by lasso
X_selected <- X[,features_by_lasso]
data_selected <- cbind(y, X[,features_by_lasso]) %>% as.data.frame()

##### ----------------------- Fit a linear model from data_selected by lasso  ----------------------- 
fit.selected <- lm(y~., data=data_selected)
(coef(fit.selected))




##### ----------------------- Stepwise feature selection, using AIC ----------------------- 
library(caret)

train.control <- trainControl(method = "cv", number = 20)

# http://topepo.github.io/caret/train-models-by-tag.html#generalized-linear-model
regfit.fwd <-  caret::train(X, as.numeric(y), method = "lmStepAIC", # "leapForward", "leapSeq"
                            trControl = train.control, intercept = FALSE)
summary(regfit.fwd)




##### ----------------------- Random forest variable importance ----------------------- 
# -------------------------- (Slow for A2) -----------------------
library( randomForest )
model <- randomForest(X, y, ntree=100, importance=T, 
                      keep.forest=T, replace = F, 
                      sampsize = 0.2*nrow(X))
plot(model)
varImpPlot(model, type=1, main = "% of increasing in MSE, Dataset A2")


##### -----------------------  Calculate Relative Importance, by "lmg" ----------------------- 
library(relaimpo)
metrics <- calc.relimp(fit.all,type=c("lmg"), rela=T)
metrics




#####################################################################################
##### -----------------------  Fit the final model, and compute MSE ##### -----------------------  
#####################################################################################
print("------------Coefficients by lm()-------------")
(coef.allVar <- coef(fit.all))

(coef.selectedVar <- coef(fit.selected))



