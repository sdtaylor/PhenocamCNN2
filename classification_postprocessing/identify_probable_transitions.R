library(tidyverse)

source('classification_postprocessing/postprocessing_tools.R')


max_gap_size_for_filling = 5

predictions = read_csv('./data/vgg16_v1_60epochs_predictions.csv') %>%
  prep_prediction_dataframe() %>%
  fill_date_gaps_with_na() %>%
  prediction_df_wide_to_long() %>%  
  fill_blurry_dates_with_na() %>%
  gap_fill_na_predictions(max_gap_size = max_gap_size_for_filling)
