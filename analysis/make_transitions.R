library(tidyverse)

#-----------------
# This will detect transitions by identifying the dates which a camera
# moved from 1 state to another defined below.
# eg. movign from residue -> soil = tillage
#-----------------

transition_signatures = tribble(
  ~category,         ~prior_day_class, ~class,     ~transition,
  'dominant_cover',  'vegetation',     'soil',     'harvest-to-soil',
  'dominant_cover',  'vegetation',     'residue',  'harvest-to-residue',
  'dominant_cover',  'residue',        'soil',     'tillage',
  'crop_status',     'no_crop',        'emergence','emergence' # this will identify the 1st day of emergence
)

final_predictions = read_csv('./data/final_predictions.csv')

final_predictions = final_predictions %>%
  pivot_longer(c(dominant_cover, crop_type, crop_status), names_to = 'category', values_to = 'class')

final_predictions_offset = final_predictions %>%
  mutate(date = date + lubridate::days(1)) %>%
  select(phenocam_name, date, category, prior_day_class = class)

transitions = final_predictions %>%
  left_join(final_predictions_offset, by=c('phenocam_name','date','category')) %>%
  left_join(transition_signatures, by=c('category','prior_day_class','class')) %>%
  filter(!is.na(transition)) %>%
  mutate(doy = lubridate::yday(date))

write_csv(transitions,'./data/transition_dates.csv')
