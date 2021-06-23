library(tidyverse)
source('classification_postprocessing/postprocessing_tools.R')
source('config.R')
"
This takes the output from a keras image classificaiton model and prepares it for post-processing
in a hidden markov model (HMM). 
"

MAX_SNOW_DAY_FILL_SIZE = 60
MAX_MISSING_OR_BLURRY_FILL_SIZE = 3
MIN_SEQUENCE_SIZE = 60

image_predictions = read_csv('./data/vgg16_v4_20epochs_predictions.csv') %>%
  prep_prediction_dataframe() %>% 
  group_by(phenocam_name, date) %>%  # for any phenocam_name,date where there is > 1 image just pick the 1st one.
  slice_head(n=1) %>%
  ungroup() %>%
  prediction_df_wide_to_long() %>%
  process_snow_days() %>% 
  gap_fill_na_predictions(max_gap_size = MAX_SNOW_DAY_FILL_SIZE) %>% 
  fill_blurry_dates_with_na() %>%
  remove_blurry_class_and_normalize() %>%
  prediction_df_long_to_wide()  %>%
  fill_date_gaps_with_na %>%
  prediction_df_wide_to_long() %>%
  gap_fill_na_predictions(max_gap_size = MAX_MISSING_OR_BLURRY_FILL_SIZE) 
  
  
image_predictions = image_predictions %>%
  assign_sequence_identifiers() %>%
  filter(!is.na(site_sequence_id)) %>%
  group_by(phenocam_name, category, site_sequence_id) %>%
  filter(n_distinct(date) > MIN_SEQUENCE_SIZE) %>%
  ungroup()

image_predictions %>%
  filter(category=='dominant_cover') %>%
  pivot_wider(names_from = 'class', values_from='probability') %>%
  write_csv('data/image_predictions_for_hmm-dominant_cover.csv')

image_predictions %>%
  filter(category=='crop_status') %>%
  pivot_wider(names_from = 'class', values_from='probability') %>% 
  write_csv('data/image_predictions_for_hmm-crop_status.csv')

# crop type is not run thru the HMM, but the probabilities are used in final processing
image_predictions %>%
  filter(category=='crop_type') %>%
  write_csv('data/image_predictions_for_final_processing-crop_type.csv')
