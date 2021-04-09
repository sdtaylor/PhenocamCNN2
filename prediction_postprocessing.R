library(tidyverse)
#library(patchwork)
#library(ggnewscale)
library(zoo)

#source('config.R')

max_gap_size_for_filling = 5

predictions = read_csv('./data/vgg16_v1_60epochs_predictions.csv') %>%
  mutate(file = str_remove(file, '.jpg')) %>%
  separate(file, c('phenocam_name','datetime'), '_', extra='merge') %>%
  mutate(datetime = str_replace_all(datetime,'_','-')) %>%
  mutate(datetime = lubridate::as_datetime(datetime, format='%Y-%m-%d-%H%M%S')) %>%
  mutate(year = lubridate::year(datetime),
         date = lubridate::date(datetime)) %>%
  select(-filepath,-datetime) %>%
  arrange(date)


fill_date_gaps_with_na = function(df){
  # where there are non-continuous timeseries, insert missing dates with NA's for all probabilites
  full_date_range = tibble(date = seq(min(df$date), max(df$date), 'day'))
  
  df %>%
    full_join(full_date_range, by='date') %>%
    arrange(date)
}

predictions2 = predictions %>%
  group_by(phenocam_name) %>%
  do(fill_date_gaps_with_na(.)) %>%
  ungroup()

predictions = predictions %>%
  #filter(phenocam_name=='cafboydnorthltar01',year==2020) %>%
  pivot_longer(c(-phenocam_name,-date,-year), names_sep='-', names_to=c('category','class'), values_to='probability')

#------------------------------------
# if blurry is the highest probability on a given date, 
# mark that entire date as NA
#-----------------------------------

# Ensure no NA probabilites exist before introducing some
#if(sum(is.na(predictions$probability))) stop('NA probabilites in prediction probabilites dataset')

blurry_dates = predictions %>%
  filter(category == 'dominant_cover') %>% # if this category has a blurry photo then everything for this date is blurry
  group_by(phenocam_name, date) %>%
  slice_max(probability) %>%
  ungroup() %>%
  filter(class=='blurry') %>%
  mutate(is_blurry = TRUE) %>%
  select(phenocam_name, date, is_blurry)

predictions = predictions %>%
  left_join(blurry_dates, by=c('phenocam_name','date')) %>%
  mutate(is_blurry = replace_na(is_blurry, FALSE)) %>%
  mutate(probability = ifelse(is_blurry, NA, probability)) %>%
  select(-is_blurry)

predictions = predictions %>%
  mutate(was_na = is.na(probability)) %>%
  group_by(phenocam_name, category, class) %>%
  mutate(probability = zoo::na.approx(probability, maxgap=max_gap_size_for_filling)) %>%
  ungroup()

# Visualz some gap filling 
# predictions  %>%
#   filter(phenocam_name=='arsmorris1', year==2018, category == 'dominant_cover') %>%
#   ggplot(aes(x=date, y=probability)) +
#   geom_line() +
#   geom_point(aes(color=was_na), size=2) +
#   facet_wrap(~class)

#------------------------------------
# Apply abrubt transition detection
#-----------------------------------
x = predictions %>%
  filter(phenocam_name=='arsmorris1', category=='dominant_cover', class=='vegetation') 

apply_harvest_detection = function(probabilites, threshold = 0.5){
  n = length(probabilites)
  if(n<5) stop('harvest detection needs a window size >=5')
  if(n%%2==0) stop('harvest detection window should be odd')
  midpoint = n-floor(n/2)
  starting_mean = mean(probabilites[1:(midpoint-1)])
  ending_mean   = mean(probabilites[(midpoint+1):n])
  return(starting_mean - ending_mean >= 0.5)
}

harvest_detection_window = 7
harvest_detection_threshold = 0.5

x$harvest = zoo::rollapply(x$probability, 
               width=harvest_detection_window,
               FUN = apply_harvest_detection,
               threshold = harvest_detection_threshold,
               fill=NA)

ggplot(x, aes(x=date, y=probability)) + 
  geom_line() + 
  geom_point(aes(color=harvest), size=4)
