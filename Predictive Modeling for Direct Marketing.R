# Using preditive model, the expected profit should be 1000*0.13*(1-FNR)*40-1000*0.87*FPR*10 = 5200-5200*FNR-8700*FPR

# read in and prepare the data
install.packages('rpart')
install.packages('caret')
library(rpart)
library(caret)
bank <- read.csv("~/bank.csv")
bank <- data.frame(bank)  # convert from tibble to data frame
str(bank)

# convert some strings into factors
bank$default <- as.factor(bank$default)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$y <- as.factor(bank$y)

# split the data set
set.seed(644)   # for reproducible results
train <- sample(1:nrow(bank),0.66667*nrow(bank))
b.train <- bank[train,]
b.test <- bank[-train,]

# build a basic big tree with the training date using minsplit = 10 and cp = 0
library(rpart)
fit <- rpart(y ~ ., 
             data = b.train,
             method = "class",
             control = rpart.control(minsplit = 10, cp = 0))
fit

# Find the CP which provides the lowest error
bestcp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
bestcp
#   build a pruned tree with best cp
fit.best_cp <- rpart(y ~ ., 
                   data = b.train,
                   control=rpart.control(minsplit = 10, cp=bestcp))
# draw the pruned tree
plot(fit.best_cp, uniform=T, branch=0.5, compress=T,
     main="Tree with best cp", margin=0.05)
text(fit.best_cp,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2)

# count the number of nodes of the pruned tree
nrow(fit.best_cp$frame)
#   from the pruned tree (fit.best_cp) we can find that the total number of nodes is 23

# creat the confusion matrix for the test data set and the tree
library(caret)
#   for test data set
confusion.test <- confusionMatrix(table(pred = predict(fit.best_cp, b.test, type="class"),
                                         actual = b.test$y), positive="yes")
#   for the pruned tree
confusion.train <- confusionMatrix(table(pred = predict(fit.best_cp, b.train, type="class"),
                                        actual = b.train$y), positive="yes")

# find the error rates for the test data
test.error_rates <- as.numeric(1 - confusion.test$overall["Accuracy"])                          
test.FNR <- 1 - confusion.test$byClass["Sensitivity"][[1]]                                     
## confusion.test$table[1,2]/(confusion.test$table[1,2] + confusion.test$table[2,2])
test.FPR <- 1 - confusion.test$byClass["Specificity"][[1]]                                    
## confusion.test$table[2,1]/(confusion.test$table[2,1] + confusion.test$table[1,1])

# compute the expected profit according to the test.FNR and test.FPR
test.expected_profit <- 5200-5200*test.FNR-8700*test.FPR      # profit = 1500.917

# recreat a new balanced training data set
b.train.yes <- b.train[b.train$y == "yes", ]
b.train.no <- b.train[b.train$y == "no", ]
set.seed(234)                                          
sub <- sample(1:nrow(b.train.no),nrow(b.train.yes))  
sub.b.train.no <- b.train.no[sub, ]                    
b.bal <- rbind(b.train.yes, sub.b.train.no)       

# build a tree with the balanced training date using minsplit = 10 and cp = 0
fit.bal <- rpart(y ~ ., 
             data = b.bal,
             method = "class",
             control = rpart.control(minsplit = 10, cp = 0))
fit.bal

#   a. find the CP which provides the lowest error
bestcp.bal <- fit.bal$cptable[which.min(fit.bal$cptable[,"xerror"]),"CP"]
bestcp.bal
#   build a pruned tree with best cp
fit.best_cp.bal <- rpart(y ~ ., 
                     data = b.bal,
                     control=rpart.control(minsplit = 10, cp=bestcp.bal))
#   draw the tree
plot(fit.best_cp.bal, uniform=T, branch=0.5, compress=T,
     main="balanced Tree with best cp", margin=0.05)
text(fit.best_cp.bal,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2)

#   b. count the number of nodes of the pruned tree
fit.best_cp.bal
#   from the pruned tree (fit.best_cp) we can find that the total number of nodes is 51

#   c. creat the confusion matrix for the test data set and the tree
#   for test data set
confusion.test.bal <- confusionMatrix(table(pred = predict(fit.best_cp.bal, b.test, type="class"),
                                        actual = b.test$y), positive="yes")
#   for the pruned tree
confusion.train.bal <- confusionMatrix(table(pred = predict(fit.best_cp, b.bal, type="class"),
                                         actual = b.bal$y), positive="yes")

#   d. find the error rates for the test data
test.error_rates.bal <- as.numeric(1 - confusion.test.bal$overall["Accuracy"])                                  
test.FNR.bal <- 1 - confusion.test.bal$byClass["Sensitivity"][[1]]                                              
## confusion.test.bal$table[1,2]/(confusion.test.bal$table[1,2] + confusion.test.bal$table[2,2])
test.FPR.bal <- 1 - confusion.test.bal$byClass["Specificity"][[1]]                                              
## confusion.test.bal$table[2,1]/(confusion.test.bal$table[2,1] + confusion.test.bal$table[1,1])

#   e. compute the expected profit according to the test.FNR and test.FPR
test.bal.expected_profit <- 5200-5200*test.FNR.bal-8700*test.FPR.bal          

# find the optimal cutoff for balanced model
library(ROCR)

# predict the prob
pro.pred <- as.data.frame(predict(fit.best_cp.bal, b.bal, type="prob"))
# compute the score
pro.pred.score <- 
  prediction(pro.pred[,2],  # the predicted P[Yes]
             b.bal$y) # the actual class

# compute the performance object 
pro.pred.perf <- performance(pro.pred.score, "tpr", "fpr")

# general evaluation of ROC of classifiers
performance(pro.pred.score, "auc")@y.values  # 0.940398

# built performance according cost
pro.cost <- performance(pro.pred.score, measure="cost", 
                         cost.fn=5200, cost.fp=8700)

# draw the cost changing trend
plot(pro.cost)

# find the best point
cutoff.best <- pro.cost@x.values[[1]][which.min(pro.cost@y.values[[1]])]
cutoff.best  # 0.7142857

# predict based on best cutoff probability
b.pred.best <- predict(fit.best_cp.bal, b.test, type="prob")
b.pred.best.cutoff <- ifelse(b.pred.best[,2] <= cutoff.best,'no','yes')

confusion.mincost <- confusionMatrix(table(pred = b.pred.best.cutoff,
                                           actual = b.test$y), positive='yes')

test.FNR.cutoff <- 1 - confusion.mincost$byClass["Sensitivity"][[1]]                                       
## confusion.test.bal$table[1,2]/(confusion.test.bal$table[1,2] + confusion.test.bal$table[2,2])
test.FPR.cutoff <- 1 - confusion.mincost$byClass["Specificity"][[1]]                                       
## confusion.test.bal$table[2,1]/(confusion.test.bal$table[2,1] + confusion.test.bal$table[1,1])

# compute the expected profit according to the test.FNR and test.FPR
test.cutoff.expected_profit <- 5200-5200*test.FNR.cutoff-8700*test.FPR.cutoff           
# profit = 1981.34
test.cutoff.expected_profit



# Comparing the models built from b.train(raw training data set), b.bal(balanced training) 
# and the b.bal with cost minimizing cutoff (change default cutoff of 50% to the best cutoff),
# we can easily see that the expected profit goes up and up.
# The reason of this is that from b.train to b.bal, we reduce the sampling bias through balancing
# the numbers of positive observations and negative observations(randomly taking out some "no") and
# adjust the default 50% cutoff to the cost-minimizing cutoff, making the model better fit the real world.
