library(tidyverse)
library(zoo)

source('config.R')
# ---------------
# Partition the annual Gcc curves into rough seasons to sample
# images from.
# This creates a more representative sample of distinct crop phenologies
# ---------------

get_transition_date_file = function(f){
  str_replace(f,'_1day','_1day_transition_dates')
}

all_gcc_files = list.files(phenocam_gcc_folder, pattern = '1day.csv', full.names = TRUE)

all_site_periods = tibble()

for(full_file_path in all_gcc_files){
  site_gcc = read_csv(full_file_path, skip=24) 
  site_transition = read_csv(get_transition_date_file(full_file_path), skip = 16) %>%
    filter(gcc_value == 'gcc_90') %>%
    select(phenocam_name = site, direction, transition_10, transition_25, transition_50) %>%
    gather(threshold, date, -direction, -phenocam_name) %>%
    arrange(date) %>%
    mutate(original_transition_date = TRUE)
  
  all_dates = tibble(date = seq(min(site_transition$date), max(site_transition$date), '1 day'))
  
  # make a data.frame of *all* dates in the range
  # and fill in the direction, transition threshold
  site_transition = site_transition %>%
    full_join(all_dates, by='date') %>%
    arrange(date)
  
  site_transition$direction = zoo::na.locf(site_transition$direction)
  site_transition$threshold = zoo::na.locf(site_transition$threshold)
  site_transition$phenocam_name = zoo::na.locf(site_transition$phenocam_name)
  
  site_transition = site_transition %>%
    mutate(period = case_when(
      direction == 'falling' & threshold == 'transition_50' ~ 'senescing',
      direction == 'falling' & threshold == 'transition_25' ~ 'senescing',
      direction == 'falling' & threshold == 'transition_10' ~ 'senesced',
      direction == 'rising'  & threshold == 'transition_50' ~ 'peak',
      direction == 'rising'  & threshold == 'transition_25' ~ 'growing',
      direction == 'rising'  & threshold == 'transition_10' ~ 'growing',
    ))
  
  # Quick plot to check on this logic
  # site_transition$period = factor(site_transition$period, levels=c('senesced','growing','peak','senescing'), ordered=T)
  # ggplot(site_transition, aes(x=date, y=period, color=period)) + 
  #   geom_point()
  
  image_info = site_gcc %>%
    select(image_filename = midday_filename, date, snow_flag) %>%
    filter(str_detect(image_filename,'.jpg'))

  site_transition = site_transition %>%
    left_join(image_info, by='date')
  
  
  ########################
  ########################
  # Count the size of daily sequences which have an available image file 
  # for all days. And label each sequence with a unique id
  site_transition$running_tally = NA
  site_transition$site_seq_id = NA
  tally=0
  seq_id=0
  for(i in 1:nrow(site_transition)){
    
    current_row_present = !is.na(site_transition$image_filename[i])
    if(current_row_present){
      tally = tally + 1
    } else {
      tally = 0
    }
    
    if(tally==1){
      seq_id = seq_id+1
      site_transition$site_seq_id[i] = seq_id
    } else if(tally>1){
      site_transition$site_seq_id[i] = seq_id
    }
    
    site_transition$running_tally[i] = tally
  }
  
  ######
  all_site_periods = all_site_periods %>%
    bind_rows(site_transition)
}


# for each site, pick at most X random images from each period type
# this is for the training data.
images_for_download = all_site_periods %>%
  filter(!is.na(image_filename)) %>%
  group_by(phenocam_name, period) %>%
  sample_n(min(n(),random_images_per_period))

write_csv(images_for_download, training_images_download_list)

# save the *full* image list to create the final timeseries products
all_site_periods %>%
  filter(!is.na(image_filename)) %>%
  write_csv(full_image_download_list)

