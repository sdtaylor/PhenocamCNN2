library(tidyverse)

source('classification_postprocessing/postprocessing_tools.R')
source('config.R')

# Here the final processing is done to combine crop_type and the hmm processed
# dominant_cover and crop_status
#
# crop_type is most discernible with mature plants, so for every continuous
# time chunk of a single plant (as specified by crop_status!='no_crop') 
# assign a single crop_type of the highest probability for the entire chunk.
# Weighted toward the later stages

hmm_predictions = read_csv('./data/hmm_output.csv') %>%
  select(-site_sequence_id)

crop_type_predictions = read_csv('./data/image_predictions_for_final_processing-crop_type.csv') %>%
  select(phenocam_name, date, crop_class = class, probability)

# for each phenocam identify each unique crop sequence
# crop_sequence_id line from adapted from https://stackoverflow.com/a/42734207/6615512
unique_crop_segments = hmm_predictions %>%
  pivot_wider(names_from = 'category', values_from = 'hmm_class') %>%
  mutate(crop_present = crop_status != 'no_crop') %>%
  filter(crop_present) %>%
  group_by(phenocam_name) %>%
  mutate(crop_sequence_id = cumsum( c(1,diff(date)) !=1  )) %>%
  ungroup() %>%
  select(phenocam_name, date, crop_sequence_id)

sequence_crop_types = crop_type_predictions %>%
  left_join(unique_crop_segments, by=c('phenocam_name','date')) %>%
  filter(!is.na(crop_sequence_id)) %>%
  group_by(phenocam_name, crop_class, crop_sequence_id) %>%
  summarise(total_probability = sum(probability)) %>%
  ungroup() %>%
  group_by(phenocam_name, crop_sequence_id) %>%
  slice_max(total_probability) %>%
  ungroup()

# sanity check, 1 crop per phenocam/crop_sequence_id
if(!all(count(sequence_crop_types, phenocam_name, crop_sequence_id)$n)==1) stop('more than 1 crop type in some phenocam/crop_sequence_id combinations')

final_crop_types = unique_crop_segments %>%
  left_join(sequence_crop_types, by=c('phenocam_name','crop_sequence_id')) %>%
  select(phenocam_name, date, hmm_class=crop_class) %>% # not technically an hmm_class, but it should match the column name for the rest
  mutate(category='crop_type',
         year = lubridate::year(date))

final_output = hmm_predictions %>%
  bind_rows(final_crop_types) %>%
  pivot_wider(names_from = 'category', values_from = 'hmm_class') %>%
  mutate(crop_type = ifelse(crop_status=='no_crop','no_crop', crop_type)) # mark crop_type='no_crop' wherever crop_status was 'no_crop'

write_csv(final_output, final_prediction_file)
