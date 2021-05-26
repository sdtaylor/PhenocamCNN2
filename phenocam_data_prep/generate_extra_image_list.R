library(tidyverse)
library(rvest)
library(phenocamapi)

# remotes::install_github('phenocam/phenocamapi')

source('config.R')

#--------------------
# During the annotation process dates are randomly selected, and the mid-day image
# is the final one used. But phenocams take multiple images every day (up to every 30 min during daylight).
# So there is the potential to use that 1 mid-day annoation and assign the labels from it to multiple images
# from the same day. This increases the sample size and introduces different lighting conditions.
#
# Here I take the original annoation list of mid-day images and collect images from the same day for use in model fitting.


#----------------
# Some functions for scraping the image lists
#----------------

make_daily_image_url = function(phenocam_name, date){
  base_url = 'https://phenocam.sr.unh.edu/webcam/browse'
  date  = as.Date(date)
  year  = lubridate::year(date)
  month = lubridate::month(date)
  day   = lubridate::day(date)
  
  month = ifelse(month<10, paste0('0',month),month)
  day   = ifelse(day<10, paste0('0',day),day)
  
  return(paste(base_url,phenocam_name,year,month,day, sep='/'))
}

get_image_list_for_date = function(phenocam_name, date){
  image_base_url = 'https://phenocam.sr.unh.edu'
  daily_image_url = make_daily_image_url(phenocam_name, date)
  print(daily_image_url)
  daily_image_html = rvest::read_html(daily_image_url)
  
  all_links = daily_image_html %>%
    rvest::html_elements('a') %>%
    rvest::html_attr('href')
  
  all_links = all_links[stringr::str_detect(all_links, stringr::regex('/data/archive*'))]
  all_links = all_links[stringr::str_detect(all_links, stringr::regex('jpg|jpeg'))]
  if(length(all_links)==0){
    return(c())
  } else {
    all_links = paste0(image_base_url,all_links)
    return(all_links) 
  }
}


#------------------------------
training_images = read_csv(training_images_download_list)

all_daily_images = c()

for(i in 1:nrow(training_images)){
#for(i in sample(8000, 200)){
  phenocam_name_i = training_images$phenocam_name[i]
  date_i          = training_images$date[i]
  
  site_date_images = get_image_list_for_date(phenocam_name = phenocam_name_i,
                                             date          = date_i)
  
  all_daily_images = c(all_daily_images, site_date_images)
}


# this parse function will return a nice data.frame of all everthing
x = phenocamapi::parse_phenocam_filenames(all_daily_images) %>%
  select(url = filepaths,
         filenames, 
         phenocam_name = Site,
         year = Year, 
         month = Month,
         day = Day,
         hour = Hour, 
         date = Date)

# only midday-ish images. timestamps for images represent local time
x = x %>%
  filter(hour %in% extra_images_hours_to_keep)

write_csv(x, extra_image_list)

# now match the labels from the annotated images









