library(tidyverse)
library(phenocamr)

source('config.R')
# ---------------
# Collect info on all the phenocam AG cameras
# ---------------

phenocam_sites = phenocamr::list_rois() %>%
  select(phenocam_name = site, lat, lon, roi_type=veg_type, roi_id =roi_id_number, first_date, last_date, site_years) 

phenocam_sites = phenocam_sites %>%
  filter(roi_type == 'AG',
         site_years > 2) %>%
  filter((roi_id %% 1000) == 0) # No experimental ROIs, which usually end in X001

phenocam_sites = phenocam_sites %>%
  filter(!phenocam_name %in% ag_sites_to_exclude)

write_csv(phenocam_sites, phenocam_site_file)
