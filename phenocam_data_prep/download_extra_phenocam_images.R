library(tidyverse)

source('config.R')

# ---------------
# Download all the "extra" images.
# ---------------

images_for_download = read_csv(extra_image_list)

failed_downloads = tibble(phenocam_name=c(),date=c(),image_filename=c())

for(image_i in 1:nrow(images_for_download)){
  phenocam_name = images_for_download$phenocam_name[image_i]
  
  year = lubridate::year(images_for_download$date[image_i])
  
  month = lubridate::month(images_for_download$date[image_i])
  month = ifelse(month<10, paste0('0',month), as.character(month))
  
  image_filename = images_for_download$filenames[image_i]
  
  download_url = images_for_download$url[image_i]
  dest_path    = paste0(extra_training_image_folder, image_filename)
  
  download_attempt = try(download.file(url = download_url,destfile = dest_path,))
  
  if(class(download_attempt) == 'try-error'){
    failed_downloads = failed_downloads %>%
      add_row(phenocam_name = phenocam_name,
              date          = images_for_download$date[image_i],
              image_filename = image_filename)
  }
  
}
