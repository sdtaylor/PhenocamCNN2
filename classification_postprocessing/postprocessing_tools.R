
# Initial processing ----


# extract the phenocame name, date, year, etc in the file output by apply_keras_model.py
prep_prediction_dataframe = function(df){
  if(!prediction_df_is_wide(df)) stop('prediction data.frame must be wide format for initial prep')
  df %>%
    mutate(file = str_remove(file, '.jpg')) %>%
    separate(file, c('phenocam_name','datetime'), '_', extra='merge') %>%
    mutate(datetime = str_replace_all(datetime,'_','-')) %>%
    mutate(datetime = lubridate::as_datetime(datetime, format='%Y-%m-%d-%H%M%S')) %>%
    mutate(year = lubridate::year(datetime),
           date = lubridate::date(datetime)) %>%
    select(-filepath,-datetime) %>%
    arrange(date)
}

# Helper function for below.
.fill_date_gaps_with_na_single_site = function(df){
  # where there are non-continuous timeseries, insert missing dates with NA's for all probabilites
  full_date_range = tibble(date = seq(min(df$date), max(df$date), 'day'))
  
  df %>%
    full_join(full_date_range, by='date') %>%
    arrange(date)
}

# For every site ensure there is a continuous timeseries from beginning to end.
# This may not be the case due to missing images, so fill them with NA.
fill_date_gaps_with_na = function(df){
  if(!prediction_df_is_wide(df)) stop('prediction data.frame must be wide format filling date gaps')
  df %>%
    group_by(phenocam_name) %>%
    do(.fill_date_gaps_with_na_single_site(.)) %>%
    ungroup()
}



# The classifier does not attempt to classify blurry photos. Here those dates
# are filled with NA instead.
fill_blurry_dates_with_na = function(df){
  if(!prediction_df_is_long(df)) stop('prediction data.frame must be long format for blurry date filling')
  blurry_dates = df %>%
    filter(category == 'dominant_cover') %>% # if this category has a blurry photo then everything for this date is blurry
    group_by(phenocam_name, date) %>%
    slice_max(probability) %>%
    ungroup() %>%
    filter(class=='blurry') %>%
    mutate(is_blurry = TRUE) %>%
    select(phenocam_name, date, is_blurry)
  
  df %>%
    left_join(blurry_dates, by=c('phenocam_name','date')) %>%
    mutate(is_blurry = replace_na(is_blurry, FALSE)) %>%
    mutate(probability = ifelse(is_blurry, NA, probability)) %>%
    select(-is_blurry)
}

# Interpolate missing dates with surrounding probabilites for each class/category.
# If a gap is greater than max_gap_size than it will be left as NA
gap_fill_na_predictions = function(df, max_gap_size){
  if(!prediction_df_is_long(df)) stop('prediction data.frame must be long format for gap filling')
  df %>%
    mutate(was_na = is.na(probability)) %>%
    group_by(phenocam_name, category, class) %>%
    mutate(probability = zoo::na.approx(probability, maxgap=max_gap_size)) %>%
    ungroup()
}

# Helper function for below.
.assign_sequence_identifier_single_site = function(df){
  full_date_range = seq(min(df$date), max(df$date), 'day')
  if(!all(unique(df$date) %in% full_date_range)) stop('Not all dates in df for site: ',unique(df$phenocam_name))
  
  df$running_tally = NA
  df$site_sequence_id = NA
  present_tally=0
  seq_id=0
  for(i in 1:nrow(df)){
    
    current_row_present = !is.na(df$`dominant_cover-vegetation`[i])
    if(current_row_present){
      present_tally = present_tally + 1
      missing_tally = 0
    } else {
      present_tally = 0
    }
    
    if(present_tally==1){
      seq_id = seq_id+1
      df$site_sequence_id[i] = seq_id
    } else if(present_tally>1){
      df$site_sequence_id[i] = seq_id
    }
    
    df$running_tally[i] = present_tally
  }
  
  df %>%
    select(-running_tally)
}

# For each location assign identifiers to contiguous chunks (ie. non NA) timeseries
assign_sequence_identifiers = function(df){
  if(!prediction_df_is_wide(df)) stop('prediction data.frame must be wide format for sequence identification')
  
  df %>%
    group_by(phenocam_name) %>%
    do(.assign_sequence_identifier_single_site(.)) %>%
    ungroup()
  
}

# Reshaping Tools ----
# Working with the predictions data.frame requires some back and forth between wide/long formats.

prediction_df_is_long = function(df){
  all(c('category','class') %in% colnames(df))
}

prediction_df_is_wide = function(df){
 !prediction_df_is_long(df) 
}

# Go to a long data.frame with 'category','class','probability' columns.
prediction_df_wide_to_long = function(df){
  df %>%
    pivot_longer(c(-phenocam_name,-date,-year,-starts_with('transition-')), names_sep='-', names_to=c('category','class'), values_to='probability')
}

# Go to a wide data.frame with columns for every category/class. eg. columns of 'dominant_cover-vegetation','dominant_cover-residue', etc.
prediction_df_long_to_wide = function(df){
  df %>%
    pivot_wider(names_from=c('category','class'), names_sep='-', values_from='probability')
}

# Transition date detection ----
# Used in transition date dection, calculate the average change in probability 
#for a single time window from the 1st to the 2nd half. 
calculate_mean_change = function(probabilites){
  #probabilites = as.numeric(probabilites)
  n = length(probabilites)
  if(n<5) stop('change detection needs a window size >=5')
  if(n%%2==0) stop('change detection window should be odd')
  midpoint = n-floor(n/2)
  starting_mean = mean(probabilites[1:(midpoint-1)])
  ending_mean   = mean(probabilites[(midpoint+1):n])
  return(ending_mean - starting_mean)
}

# harvest detection. vegetation probability must decrease while soil + residue increase.
# soil and residue look really similar and are commonly misclassified as eachother, 
# so detecting when their sum exceeds some threshold is more viable. 
# Note this cannot be just when vegetation decreases since it might be snowed it, which is not actually harvest.
apply_harvest_detection = function(d, veg_threshold, soil_residue_threshold){
  # veg_threshold should be < 0 for decrease
  # soil_residue_threshold should be > 0 to signify increase
  d = as_tibble(d)
  d = arrange(d, date)
  veg_change          = calculate_mean_change(as.numeric(d$`dominant_cover-vegetation`))
  soil_residue_change = calculate_mean_change(as.numeric(d$`dominant_cover-soil`) + as.numeric(d$`dominant_cover-residue`))
  return((veg_change <= veg_threshold) & (soil_residue_change >= soil_residue_threshold))
}

## Transition date detection example ----
# An example of how to use the above functions
# where the predictions data.frame has columns c('phenocam_name','date','category','class','probability')
if(FALSE){
  harvest_detection_window = 7
  harvest_detection_veg_threshold = -0.2
  harvest_detection_soil_residue_threshold = 0.2
  
  x = predictions %>%
    filter(phenocam_name %in% c('mandanh5','arsmorris1')) %>%
    prediction_df_long_to_wide() %>%
    group_by(phenocam_name) %>%
    nest() %>%
    mutate(`transition-harvest` = map(data, ~zoo::rollapply(data  = .,
                                                            FUN                    = apply_harvest_detection,
                                                            width                  = harvest_detection_window,
                                                            veg_threshold          = harvest_detection_veg_threshold,
                                                            soil_residue_threshold = harvest_detection_soil_residue_threshold,
                                                            by.column = F,
                                                            fill=NA))) %>%
    unnest(cols=c('data',starts_with('transition-'))) %>%
    mutate(`transition-harvest` = replace_na(`transition-harvest`, F)) # NA's happen at the end/beginning of continuous timeseries.
}