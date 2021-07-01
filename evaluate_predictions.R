library(tidyverse)

source('classification_postprocessing/postprocessing_tools.R')
source('config.R')

#-------------------------
# Prep model output ----

hmm_predictions = read_csv('./data/final_predictions.csv') %>%
  pivot_longer(c('dominant_cover','crop_status','crop_type'), names_to='category', values_to='hmm_class')

original_predictions = read_csv('./data/vgg16_v4_20epochs_predictions.csv') %>%
  mutate(filename=file) %>%
  prep_prediction_dataframe() %>% 
  #group_by(phenocam_name, date) %>%  # for any phenocam_name,date where there is > 1 image just pick the 1st one. There are only 1-2 instances of this with ~2/day
  #slice_head(n=1) %>%
  #ungroup() %>%
  prediction_df_wide_to_long()

max_probability = original_predictions %>%
  group_by(phenocam_name, filename, year, date, category, class) %>% # deal with >1 photo per day
  summarise(probability = mean(probability), n=n()) %>%
  ungroup() %>%
  group_by(phenocam_name, filename, year, date, category) %>%
  slice_max(probability, n=1) %>%
  ungroup() %>%
  select(phenocam_name, filename, date, category, maxprob_class = class)

#-------------------------
# Prep original training data ----
val_train_split_info = read_csv('./data/val_train_split_info.csv') 

class_descriptions = read_csv('./train_image_annotation/image_classes.csv')

original_training_data = read_csv(train_image_annotation_file) %>%
  mutate(filepath = NA,
         filename = file) %>%
  # left_join(val_train_split_info, by='file') %>%
  # filter(is_validation) %>%
  select(-time) %>% # leftover from annotation
  prep_prediction_dataframe() 

# Load the extra images and annotate using the hand annotated data
extra_image_training_data = read_csv('data/extra_images_for_fitting.csv') %>%
  select(phenocam_name, date, filename) %>%
  left_join(select(original_training_data,-filename), by=c('phenocam_name','date')) %>%
  filter(complete.cases(.)) # some extra images which don't have a date in the original_training_data may have snuck in

all_training_data = original_training_data %>%
  bind_rows(extra_image_training_data) %>% 
  inner_join(val_train_split_info, by='filename') %>%
  distinct() %>%   # this distinct drops any duplicates, inevitable with 100k+ images
  pivot_longer(c('crop_status','dominant_cover','crop_type'), names_to='category', values_to='class_value') %>%
  left_join(class_descriptions, by=c('category','class_value')) %>%
  select(phenocam_name, filename, year, date, category, is_validation, true_class = class_description)


#-------------------------

# possible values from caret confustion matrix:
# 'sensitivity','specificity','pos_pred_value','neg_pred_value','precision','recall','f1',
# 'prevalence','detection_rate','detection_prevalence','balanced_accuracy'
get_stats = function(d, stats = c('f1','balanced_accuracy')){
  all_classes = unique(c(d$true_class, d$predicted_class))
  d$true_class = factor(d$true_class, levels = all_classes)
  d$predicted_class = factor(d$predicted_class, levels = all_classes)
  x = caret::confusionMatrix(data = d$predicted_class, reference = d$true_class)
  
  x %>% 
    broom::tidy() %>% 
    select(term, class, estimate) %>% 
    filter(class %in% all_classes) %>% 
    filter(term %in% stats)
  
}

all_data = all_training_data %>%
  left_join(select(max_probability,filename,category,maxprob_class), by=c('filename','category'))

prediction_stats = all_data %>%
  rename(predicted_class = maxprob_class) %>%
  group_by(category, is_validation) %>%
  nest() %>%
  mutate(stats = map(data, ~get_stats(., stats=c('sensitivity','specificity','f1')))) %>%
  select(-data) %>%
  unnest(stats) %>%
  pivot_wider(names_from = 'term', values_from='estimate')



# blurry images are exluded from the final hmm model. their original annotation category was just "unknown", which fyi is
# distinct from "unknown_plant" in the crop type.
original_training_data = original_training_data %>%
  filter(true_class != 'unknown') %>%
  mutate(true_class)

# Compare them ----
library(caret)

all_data = original_training_data %>%
  left_join(max_probability, by=c('phenocam_name','date','category')) %>%
  left_join(hmm_predictions, by=c('phenocam_name','date','category')) %>%
  pivot_longer(c('true_class','maxprob_class','hmm_class'), names_to='source',values_to='class') %>%
  mutate(class = factor(class)) %>%    # everything should be a factor with the same levels
  #mutate(class = recode(class, fallow = 'other')) %>%
  pivot_wider(names_from = 'source', values_from='class') %>%
  filter(complete.cases(.))

# Checking out some of the coverage differences between the full and training datasets
# training_dates = all_data %>% 
#   mutate(month = lubridate::month(date)) %>%
#   group_by(phenocam_name, year, month) %>%
#   summarise(n_training_dates = n_distinct(date)) %>%
#   ungroup() 
#
# prediction_dates = predictions %>%
#   mutate(month = lubridate::month(date)) %>%
#   group_by(phenocam_name, year, month) %>%
#   summarise(n_prediction_dates = n_distinct(date)) %>%
#   ungroup()
# 
# prediction_dates %>%
#   left_join(training_dates, by=c('phenocam_name','year','month')) %>%
#   mutate(n_training_dates = replace_na(n_training_dates,0),
#          diff = n_prediction_dates - n_training_dates) %>%
#   View()



caret::confusionMatrix(data= all_data$maxprob_class, reference = all_data$true_class)
caret::confusionMatrix(data= all_data$hmm_class, reference = all_data$true_class)
caret::multiClassSummary(data= select(all_data, pred=hmm_class, obs=true_class))

# possible values from caret confustion matrix:
# 'sensitivity','specificity','pos_pred_value','neg_pred_value','precision','recall','f1',
# 'prevalence','detection_rate','detection_prevalence','balanced_accuracy'
get_stats = function(d, stats = c('f1','balanced_accuracy')){
  all_classes = unique(c(d$true_class, d$predicted_class))
  d$true_class = factor(d$true_class, levels = all_classes)
  d$predicted_class = factor(d$predicted_class, levels = all_classes)
  x = caret::confusionMatrix(data = d$predicted_class, reference = d$true_class)
  
  x %>% 
    broom::tidy() %>% 
    select(term, class, estimate) %>% 
    filter(class %in% all_classes) %>% 
    filter(term %in% stats)
  
}

# For testing
# d = all_data %>%
#   pivot_longer(c('maxprob_class','hmm_class'), names_to='prediction_type',values_to='predicted_class') %>%
#   filter(category=='dominant_cover', prediction_type=='maxprob_class')

sample_sizes = all_data %>%
  count(category, true_class, name='support') %>%
  rename(class = true_class)

prediction_stats = all_data %>%
  pivot_longer(c('maxprob_class','hmm_class'), names_to='prediction_type',values_to='predicted_class') %>%
  group_by(category, prediction_type) %>%
  nest() %>%
  mutate(stats = map(data, ~get_stats(., stats='f1'))) %>%
  select(-data) %>%
  unnest(stats)


stats1 = prediction_stats %>%
  pivot_wider(names_from = 'prediction_type',values_from='estimate') %>%
  left_join(sample_sizes, by=c('category','class'))
