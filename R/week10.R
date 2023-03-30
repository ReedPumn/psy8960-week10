## Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
set.seed(123)

## Data Import and Cleaning
gss_tbl <- read_spss("../data/GSS2016.sav") %>%
  as_tibble() %>%
  drop_na(HRS2) %>%
  select_if(~(mean(is.na(.)) * 100) <= 75) # This line combines several verbs on the same line, but only for added readability.

# Now I need to remove all IDKs and not clearly answered items for all columns.

