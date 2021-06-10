library(tidyverse)

source('config.R')

#-------------------------------------
# When I initially did annotation the crop_type levels were generalized, like large grass for corn/switchgrass, small grass for rice/wheat, other
# for everything else. But late in the analysis stage I realized most crop fields are corn/soybean, which are easily recognizable. So I redid *just* 
# the crop_type field on all images. Going thru 8000 images twice several months apart means there's a few inconsistencies, like what I consider
# a blurry image.
# So this script brings in the new crop type annotations and deals with those inconsistencies. Namely if I marked something as blurry or no_crop
# the first time around then set it that way here, and if I marked the crop_status as any actual stage but marked the crop_type_new as no crop, then 
# change the new one to unknown_plant.
# In the end this adjusted ~400 of the ~8500 annotated images. 
#-------------------------------------

#---------------------------------
# This commented out section is me working with the named labels to get things correct.
# I then transcribe the same code below and switched the numeric labels, which is whats needed in keras
#---------------------------------
# crop_redo = read_csv('./train_image_annotation/crop_type_redo.csv') %>%
#   mutate(crop_type = factor(crop_type, levels=0:7, labels=c('blurry','unknown_plant','corn','wheat/barley','soybean','alfalfa','other','no_crop'))) %>%
#   rename(crop_type_new = crop_type) %>%
#   mutate(crop_type_new = as.character(crop_type_new)) %>%
#   select(-time)
# 
# original = read_csv('./train_image_annotation/imageant_session2.csv') %>%
#   mutate(crop_type = factor(crop_type, levels=0:6, labels=c('blurry','unknown_plant','large_grass','small_grass','other','fallow','no_crop'))) %>%
#   mutate(crop_status = factor(crop_status, levels=0:6, labels=c('blurry','emergence','growth','flowers','senescing','senesced','no_crop'))) %>%
#   left_join(crop_redo, by='file') %>%
#   mutate(crop_type_new = case_when(
#     (crop_status %in% c('emergence','growth','flowers','senescing','senesced')) & (crop_type_new =='no_crop') ~ 'unknown_plant',
#     crop_status == 'blurry' ~ 'blurry',
#     crop_status == 'no_crop' ~ 'no_crop',
#     (crop_type == 'no_crop') & (crop_type_new == 'blurry') ~ 'no_crop',
#     TRUE ~ crop_type_new
#   ))


crop_redo = read_csv('./train_image_annotation/crop_type_redo.csv') %>%
  rename(crop_type_new = crop_type) %>%
  select(-time)

original = read_csv('./train_image_annotation/imageant_session2.csv') %>%
  left_join(crop_redo, by='file') %>%
  mutate(crop_type_new = case_when(
    (crop_status %in% c(1,2,3,4,5)) & (crop_type_new ==7) ~ 1,
    crop_status == 0 ~ 0,
    crop_status == 6 ~ 7,
    (crop_type == 6) & (crop_type_new == 0) ~ 7,
    TRUE ~ crop_type_new
  ))


original %>%
  select(-crop_type) %>%
  select(file, crop_status, crop_type=crop_type_new, dominant_cover,time) %>%
  write_csv(train_image_annotation_file)

# # Viewing the before and after as a sanity check
# original %>%
#   mutate(crop_type = factor(crop_type, levels=0:6, labels=c('blurry','unknown_plant','large_grass','small_grass','other','fallow','no_crop'))) %>%
#   mutate(crop_status = factor(crop_status, levels=0:6, labels=c('blurry','emergence','growth','flowers','senescing','senesced','no_crop'))) %>%
#   mutate(crop_type_new = factor(crop_type_new, levels=0:7, labels=c('blurry','unknown_plant','corn','wheat/barley','soybean','alfalfa','other','no_crop'))) %>%
#   count(crop_status, crop_type_new) %>%
#   View()
