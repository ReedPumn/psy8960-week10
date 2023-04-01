## Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
set.seed(123)

## Data Import and Cleaning
# This series of pipes reads our data file, removes all missing values in the criterion column, makes all values numeric, formats the data into a tibble, and removes columns with more than 75% missing data. The data were made numeric to enable later analyses in the caret package. Columns with a lot of missing data were removed to base our models on more stable sources of data.
gss_tbl <- read_spss("../data/GSS2016.sav") %>%
  filter(!is.na(HRS1)) %>%
  sapply(as.numeric) %>%
  as_tibble() %>%
  select_if(~(mean(is.na(.)) * 100) <= 75) # This line combines several verbs on the same line, but only for added readability.

## Visualization
# This series of pipes visualizes our criterion variable. I made a histogram with a smaller number of "buckets" by changing to binwidth to create a simpler figure.
(ggplot(gss_tbl, aes(x = HRS1)) +
  geom_histogram(binwidth = 10) +
  labs(x = "Workhours", y = "Count")) %>%
  ggsave("../figs/fig1.png", ., width = 1920, height = 1080, units = "px")

## Analysis
# This line randomizes the rows in our tibble to be later divided into training and test sets.
gss_random_tbl <- gss_tbl[sample(nrow(gss_tbl)), ]
# This line lets us know where to "slice" our data in half. We use this slice point to create our training data and test data.
gss_random75 <- round(nrow(gss_random_tbl) * 0.75, 0)
# This line creates our training set with 75% of our data. We do this to give our models enough data to form stable predictions to new data.
gss_train_tbl <- gss_random_tbl[1:gss_random75_tbl, ]
# This line splits our training data into 10 folds. We do this to safely evaluate the central tendency and variability of our metrics of interest.
kfolds <- createFolds(gss_train_tbl$HRS1, 10)
# This line creates our test set with 25% of our data. We do this to later test the predictive accuracy of our models.
gss_test_tbl <- gss_random_tbl[(gss_random75_tbl + 1):nrow(gss_random_tbl), ]

# This series of pipes trains our OLS Regression model. We trained this model on the "gss_train_tbl" using the indexOut = kfolds to use the datasets and hyperparameters we previously set.
OLS <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
# This line reports the results of the model. I added this line to check for abnormal results, such as rampant NAs.
OLS

# This series of pipes trains our elastic net model. It uses identical hyperparameters as the OLS Regression model due to them both being more simpler prediction models.
Enet <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
# This line reports the results of the model. I added this line to check for abnormal results, such as rampant NAs.
Enet

# This series of pipes trains our Random Forrest model. The model originally took far too long to run using defult hyperparameters. As a result, I used the tuneGrid argument to specify hyperparameters for a more manageable run time.
random_forest_gump <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "ranger",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE),
  tuneGrid = expand.grid(mtry = c(5, 50, 100), splitrule = "variance", min.node.size = 5) # Note: The kfold and LOOCV splitrules do not work, but the variance splitrule does work.
)
# This line reports the results of the model. I added this line to check for abnormal results, such as rampant NAs.
random_forest_gump

# This series of pipes trains our eXtreme Gradient Boosting model. This model also took far too long to run using default hyperparameters. As a result, I used chatGPT to learn what hyperparameters I could program using the tuneGrid argument.
EGB <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "xgbLinear", #This method was chosen over xgbTree because we are trying to make prediction in our data not in terms of decisions.
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE),
  tuneGrid = expand.grid(nrounds = c(10, 30, 50),
                         lambda = c(1, 10),
                         alpha = c(0.1, 0.5),
                         eta = c(0.1, 0.3)) # I kept the nrounds low and used only two values for lambda, alpha, and eta. This decision was chosen to balance predictive power with model run time.
)
# This line reports the results of the model. I added this line to check for abnormal results, such as rampant NAs.
EGB

# These four lines of code store our four model's R2 values when applied to the training data. I named these values to more easily create the table1_tbl.
FirstR2 <- OLS$results$Rsquared %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
SecondR2 <- max(Enet$results$Rsquared) %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
ThirdR2 <- max(random_forest_gump$results$Rsquared) %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
FourthR2 <- max(EGB$results$Rsquared) %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")

# These four lines store our four model's R2 values when applied to the test data. I named these values to more esasily create the table1_tbl.
holdout1 <- cor(predict(OLS, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0") # This produced the following warning message: "prediction from a rank-deficient fit may be misleading." This warning might have occured because two or more predictor variables are perfectly correlated. Given the number of columns in this dataset, it is fairly possible for two of them to show zero variance, subsequently producing perfect correlations.
holdout2 <- cor(predict(Enet, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
holdout3 <- cor(predict(random_forest_gump, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
holdout4 <- cor(predict(EGB, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2 %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")

## Publication
# This series of lines creates our tibble summarizing how well our models that were trained in one dataset predict data in the test dataset. Because I named each of the values for the "cv_rsq" and "ho_rsq" columns, it is really easy to create the tibble.
table1_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(FirstR2, SecondR2, ThirdR2, FourthR2),
  ho_rsq = c(holdout1, holdout2, holdout3, holdout4)
)

# Q1: How did your results change between models? Why do you think this happened, specifically?
# A: Generally speaking, the models get better in terms of predicting variance in the test data as their complexity increased. More specifically, the OLS regression and Elastic Net models are simpler than the Random Forest and eXtreme Gradient Boosting models due to their smaller number of hyperparameters to specify. I believe the Random Forest and eXtreme Gradient Boosting models feared the best because, since they are more complex models with more hyperparameters to specify, they had greater opportunities to find patterns in the data to better reduce cost.

# Q2: How did you results change between k-fold CV and holdout CV? Why do you think this happened, specifically? 
# A: As the models developed on the training data were applied to the test data, all R2 values decreased. Additionally, they all lost about the same amount of R2 values, though the Random Forest and eXtreme Gradient Boosting models faired the best. I think all models lost predictive capabilities because they were all overfitted, albeit to varying degrees, on the training data. This overfitting made generalization to new datasets result in more cost across all models.

# Q3: Among the four models, which would you choose for a real-life prediction problem, and why? Are there tradeoffs? Write up to a paragraph.
# A: For this question, I will assume that I am not going to change the models further and am only motivated to minimize cost. After all, I could always tune the models further to make them consider more hyperparameters for lower cost at the sacrifice of run time. Given those assumptions, I would choose the Random Forest model because it produced the highest R2 value when predicting data in the test set. It was not a close comparison, either; the Random Forest model had 9% more variance explained in the test data as compared to the next best model. If multiple models had near equivalence in their cost, then the argument could be made that further testing across different hyperparameters should be conducted. But because there was no near equivalence between models in terms of the test data's R2 values, I find the Random Forest model to be a clear winner. There are certainly tradeoffs to this decision with regards to communicability. The OLS Regression model, for instance, is much more communicable than the Random Forest model. As a result, it would take thoughtful conversations in an applied setting to help communicate what the model is doing and why I chose it over a model that could be more easily understood.