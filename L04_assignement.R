library(dplyr)
library(ggplot2)
library(GGally)
library(caret)

rm(list = ls()) # empty the environment

set.seed(1234)
ff = read.delim("forestfires.tsv", header = TRUE, sep = '\t')

head(ff)

ff$log_area = log10(ff$area +1) # logarithmize burn area
ff$X = factor(ff$X) # change the type of coordinates
ff$Y = factor(ff$Y)

par(mfrow = c(1,2))
ggplot(ff, aes(x = ff$area)) +
  geom_histogram()

ggplot(ff, aes(x = ff$log_area)) +
  geom_histogram()

# feature engineering - is_summer
ff$is_summer = ifelse(ff$month %in% c('jun', 'jul', 'aug'), 1, 0)
ff$is_summer = factor(ff$is_summer) # add a new categorical variable: is_summer
str(ff)

to_plot = select(ff, temp, RH, wind, rain, log_area, is_summer) %>% filter(log_area>0)
ggpairs(to_plot, mapping = aes(color = is_summer, alpha = 0.5))

# feature engineering - CatTemp: split temp into seperate categories
#CatTemp <- cut(ff$temp, breaks = c(0, 5, 10, 15, 20, 25, 30, 35))
#ff$CatTemp <- CatTemp
#head(ff)

#to_plot = select(ff, RH, wind, rain, log_area, CatTemp) %>% filter(log_area>0)
#ggpairs(to_plot, mapping = aes(color = CatTemp, alpha = 0.5))

# feature engineering - CatWind: split wind into seperate categories
CatWind <- cut(ff$wind, breaks = c(0, 4, 8, 12))
ff$CatWind <- CatWind
head(ff)

to_plot = select(ff, temp, RH, rain, log_area, CatWind) %>% filter(log_area>0)
ggpairs(to_plot, mapping = aes(color = CatWind, alpha = 0.5))

# binarizing categorical features
month = model.matrix(~ month -1, data = ff)
day = model.matrix(~day -1, data = ff)
#CatTemp = model.matrix(~CatTemp -1, data = ff)
CatWind = model.matrix(~CatWind -1, data = ff)
x = model.matrix(~ X -1, data = ff)
y = model.matrix(~ Y -1, data =ff)
is_summer = model.matrix(~ is_summer -1, data =ff)
#ff = cbind(ff, month, day, CatTemp, CatWind, x, y, is_summer)
#ff = select(ff, -X, -Y, -month, -day, -CatTemp, -CatWind, -is_summer, -area)
ff = cbind(ff, month, day, CatWind, x, y, is_summer)
ff = select(ff, -X, -Y, -month, -day, -CatWind, -is_summer, -area)

str(ff)


# splitting data
in_train = createDataPartition( y = ff$log_area, p = 0.8, list = FALSE)
ff_train = ff[in_train, ]
ff_test = ff[-in_train, ]

str(ff_train)
str(ff_test)

# centering and scaling training data for analysis
Preprocess_steps = preProcess(select(ff_train, FFMC, DMC, DC, ISI, temp, RH, wind, rain), method = c('center', 'scale'))

ff_train_proc = predict(Preprocess_steps, newdata = ff_train)
ff_test_proc = predict(Preprocess_steps, newdata = ff_test)
head(ff_train_proc)
head(ff_test_proc)

# checking for zero-variance features in training data, then remove them
nzv_train <- nearZeroVar(ff_train_proc, saveMetrics = TRUE) 
nzv_train

is_nzv_train = row.names(nzv_train[nzv_train$nzv == TRUE, ])
is_nzv_train

ff_train_proc = ff_train_proc[ , !(colnames(ff_train_proc) %in% is_nzv_train)]
str(ff_train_proc)

# identifying correlated predictors in training set, then remove them
cor_train = cor(ff_train_proc)
highly_cor = findCorrelation(cor_train, cutoff = .75)
ff_train_proc = ff_train_proc[ , -highly_cor]


# full model
full_model = train(log_area ~ ., 
                   data = ff_train_proc,
                   method = 'lm',
                   trControl = trainControl(method = 'cv', number = 10))

pred = predict(full_model, newdata = ff_test_proc)

postResample(pred = pred, obs = ff_test_proc$log_area) # view the metric RMSE
errors = data.frame(predicted = pred, 
                    observed = ff_test_proc$log_area,
                    error = pred - ff_test_proc$log_area)

ggplot(data = errors, aes(x = predicted, y = observed)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle(paste('Predicted vs observed'))

summary(full_model)
#library(klaR)
#plot(errors)
ggplot(errors, aes(x = predicted, y = error)) +
  geom_point() +
  geom_smooth() +
  xlab('Fitted values') +
  ylab('Residuals') +
  ggtitle('Residuals vs fitted values')
    

# forward model
forward_model = train(log_area ~., 
                      data = ff_train_proc,
                      method = 'leapForward',
                      tuneGrid = expand.grid(nvmax = 1:20),
                      trControl = trainControl(method = 'cv', number = 10))

attributes(forward_model) 
forward_model$bestTune # the best value of parameter nvmax used for the model

plot(forward_model)
plot(varImp(forward_model))
     

# random forest
library(randomForest)
random_forest = train(log_area ~., 
                      data = ff_train_proc,
                      method = 'rf',
                      tuneGrid = expand.grid(mtry = 1:20),
                      trControl = trainControl(method = 'cv', number = 10))
attributes(random_forest)
random_forest$bestTune

plot(random_forest)
varImpPlot(random_forest$finalModel)

# ridge model
ridge_model = train(log_area ~., 
                    data = ff_train_proc,
                    method = 'ridge',
                    tuneLength = 20,
                    trControl = trainControl(method = 'cv', number = 10))
ridge_model$bestTune

plot(ridge_model)
plot(varImp(ridge_model))

#lasso model
lasso_model = train(log_area ~., 
                    data = ff_train_proc,
                    method = 'lasso',
                    tuneLength = 20,
                    trControl = trainControl(method = 'cv', number = 10))

lasso_model$bestTune

plot(lasso_model)
plot(varImp(lasso_model))

pred = predict(lasso_model, newdata = ff_test_proc)

# neural networks model
nn_model = train(log_area ~., 
                 data = ff_train_proc,
                 method = 'brnn',
                 tuneGrid = expand.grid(neurons = 1:20),
                 trControl = trainControl(method = 'cv', number = 10))

nn_model$bestTune

plot(nn_model)
plot(varImp(nn_model))

# model comparison
results = resamples(list(full_model = full_model,
                         forward_model = forward_model,
                         random_forest = random_forest,
                         ridge_model = ridge_model,
                         lasso_model = lasso_model))
attributes(results)
dotplot(results) 
summary(results)



