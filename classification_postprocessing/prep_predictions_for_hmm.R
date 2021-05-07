library(tidyverse)
source('classification_postprocessing/postprocessing_tools.R')

"
This takes the output from a keras image classificaiton model and prepares it for post-processing
in a hidden markov model (HMM). 
"

MAX_NA_FILL_SIZE = 3
MIN_SEQUENCE_SIZE = 60

image_predictions = read_csv('./data/vgg16_v1_60epochs_predictions.csv') %>%
  prep_prediction_dataframe() %>% 
  group_by(phenocam_name, date) %>%  # for any phenocam_name,date where there is > 1 image just pick the 1st one.
  slice_head(n=1) %>%
  ungroup() %>%
  prediction_df_wide_to_long() %>%
  fill_blurry_dates_with_na() %>%
  remove_blurry_class_and_normalize() %>%
  gap_fill_na_predictions(max_gap_size = MAX_NA_FILL_SIZE) %>%
  prediction_df_long_to_wide() 
  
  
image_predictions = image_predictions %>%
  assign_sequence_identifiers() %>%
  group_by(phenocam_name, site_sequence_id) %>%
  filter(n() > MIN_SEQUENCE_SIZE) %>%
  ungroup() %>%
  prediction_df_wide_to_long()

image_predictions %>%
  filter(category=='dominant_cover') %>%
  pivot_wider(names_from = 'class', values_from='probability') %>%
  write_csv('data/image_predictions_for_hmm-dominant_cover.csv')

image_predictions %>%
  filter(category=='crop_type') %>%
  pivot_wider(names_from = 'class', values_from='probability') %>%
  write_csv('data/image_predictions_for_hmm-crop_type.csv')

image_predictions %>%
  filter(category=='crop_status') %>%
  pivot_wider(names_from = 'class', values_from='probability') %>%
  write_csv('data/image_predictions_for_hmm-crop_status.csv')