library(phenocamr)
library(tidyverse)

source('config.R')

# ---------------
# Get Gcc data to help select which images to use as training data
# ---------------

phenocam_sites = read_csv(phenocam_site_file)

failed_downloads = data.frame(site=NA, veg_type=NA)

for(i in 1:nrow(phenocam_sites)){
  download_attempt = try(download_phenocam(site = paste0(phenocam_sites$phenocam_name[i],'$'),
                    veg_type = paste0(phenocam_sites$roi_type[i]),
                    frequency = 1,
                    phenophase = TRUE,
                    out_dir = phenocam_gcc_folder))
  
  if(class(download_attempt) == 'try-error'){
    failed_downloads = failed_downloads %>%
      add_row(site=phenocam_sites$phenocam_name[i],
              veg_type = phenocam_sites$roi_type[i])
  }
}
