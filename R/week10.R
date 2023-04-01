## Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
set.seed(123)

## Data Import and Cleaning
gss_tbl <- read_spss("../data/GSS2016.sav") %>%
  filter(!is.na(HRS1)) %>%
  sapply(as.numeric) %>%
  as_tibble() %>%
  select_if(~(mean(is.na(.)) * 100) <= 75) # This line combines several verbs on the same line, but only for added readability.

## Visualization

(ggplot(gss_tbl, aes(x = HRS1)) +
  geom_histogram(binwidth = 10) +
  labs(x = "Workhours", y = "Count")) %>%
  ggsave("../figs/fig1.png", ., width = 1920, height = 1080, units = "px")

## Analysis
gss_random_tbl <- gss_tbl[sample(nrow(gss_tbl)), ]
gss_random75 <- round(nrow(gss_random_tbl) * 0.75, 0)
gss_train_tbl <- gss_random_tbl[1:gss_random75_tbl, ]
kfolds <- createFolds(gss_train_tbl$HRS1, 10)
gss_test_tbl <- gss_random_tbl[(gss_random75_tbl + 1):nrow(gss_random_tbl), ]

OLS <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
OLS

Enet <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
Enet
