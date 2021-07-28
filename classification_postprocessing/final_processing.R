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

# unknown_plant is used during emergence when it's very hard to tell the actual plant.
# With the full timeeries available, it's possible to know the crop type later
# on and thus propagate that knowledge back to the emergence stage.
crop_type_predictions = crop_type_predictions %>%
  filter(crop_class != 'unknown_plant')

# for each phenocam identify each unique crop sequence
# crop_sequence_id line from adapted from https://stackoverflow.com/a/42734207/6615512
unique_crop_segments = hmm_predictions %>%
  pivot_wider(names_from = 'category', values_from = 'hmm_class') %>%
  mutate(crop_present = crop_status != 'no_crop') %>%
  filter(crop_present) %>%
  group_by(phenocam_name) %>%
  mutate(crop_sequence_id = cumsum( c(1,diff(date)) !=1  )) %>%
  ungroup() %>%
  select(phenocam_name, date, crop_sequence_id, crop_status)

# for each phenocam and crop sequence identify the
# crop type with the highest probability across all days
# within the sequence
sequence_crop_types = crop_type_predictions %>%
  left_join(unique_crop_segments, by=c('phenocam_name','date')) %>%
  filter(!is.na(crop_sequence_id)) %>%
  group_by(phenocam_name, crop_class, crop_sequence_id) %>%
  summarise(total_probability = sum(probability)) %>%
  ungroup() %>%
  group_by(phenocam_name, crop_sequence_id) %>%
  slice_max(total_probability) %>%
  ungroup()

# the  unknown crop type was dropped above, but here we put it back for *some* instances.
# 1. for short crop sequences (<= 60 days) where the status is predominately emergence.
# 2. if the most likely crop type was "no_crop", which is unlikely since no crop would not
#    have been chosen in the crop status category for the respective sequence, thus a crop is present.
unknown_crop_identification = unique_crop_segments %>%
  group_by(phenocam_name, crop_sequence_id) %>%
  summarise(most_common_status = names(sort(table(crop_status), decreasing=T)[1]),
            n_days = n()) %>%
  ungroup() %>%
  mutate(make_crop_type_unknown = most_common_status=='emergence' & n_days <= 60) %>%
  select(phenocam_name, crop_sequence_id, make_crop_type_unknown)

sequence_crop_types = sequence_crop_types %>%
  left_join(unknown_crop_identification,  by=c('phenocam_name','crop_sequence_id')) %>%
  mutate(crop_class = ifelse(make_crop_type_unknown, 'unknown_plant', crop_class)) %>%
  mutate(crop_class = ifelse(crop_class == 'no_crop', 'unknown_plant', crop_class))

# sanity check, 1 crop type per phenocam/crop_sequence_id
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
