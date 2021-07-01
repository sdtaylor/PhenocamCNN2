library(tidyverse)
library(zoo)
library(kableExtra)

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
val_train_split_info = read_csv('./data/val_train_split_info.csv') %>%
  mutate(data_type = ifelse(is_validation, 'val','train')) %>%
  select(filename, data_type)

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
  select(phenocam_name, filename, year, date, category, data_type, true_class = class_description)


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
sample_sizes = all_data %>%
  count(category, true_class) %>%
  rename(class = true_class, sample_size=n)

prediction_stats = all_data %>%
  rename(predicted_class = maxprob_class) %>%
  group_by(category, data_type) %>%
  nest() %>%
  mutate(stats = map(data, ~get_stats(., stats=c('sensitivity','specificity','f1')))) %>%
  select(-data) %>%
  unnest(stats) %>%
  left_join(sample_sizes, by=c('category','class'))


table_row_order = tribble(
  ~category, ~class, ~rank,
  'dominant_cover', 'blurry', 1,
  'dominant_cover', 'vegetation', 2,
  'dominant_cover', 'residue', 3,
  'dominant_cover', 'soil', 4,
  'dominant_cover', 'snow', 5,
  'dominant_cover', 'water', 6,
  
  'crop_type', 'blurry', 7,
  'crop_type', 'unknown_plant', 8,
  'crop_type', 'corn', 9,
  'crop_type', 'wheat/barley', 10,
  'crop_type', 'soybean', 11,
  'crop_type', 'alfalfa', 12,
  'crop_type', 'other', 13,
  'crop_type', 'no_crop', 14,
  
  'crop_status', 'blurry', 15,
  'crop_status', 'emergence', 16,
  'crop_status', 'growth', 17,
  'crop_status', 'flowers', 18,
  'crop_status', 'senescing', 19,
  'crop_status', 'senesced', 20,
  'crop_status', 'no_crop', 21,
)

# Table formatting stuff happens here
prediction_stats %>%
  rename(metric = term, metric_value = estimate) %>%
  mutate(metric_value = case_when(
    metric=='sample_size' ~ as.character(metric_value),                     # for sample size keep the same
    TRUE ~ trimws(format(round(metric_value,2),drop0trailing=F)))) %>% # for decimals keep 2 sig. digis.
  pivot_wider(names_from = c('metric','data_type'), values_from='metric_value') %>%
  select(category, class, sample_size, starts_with('sensitivity'), starts_with('specificity'), starts_with('f1'))%>%   # column order of table
  #mutate(error_text = paste0(val,', ',train)) %>%
  #select(-train, -val) %>%
  #pivot_wider(names_from = 'metric', values_from = 'error_text') %>%
  left_join(table_row_order, by=c('category','class')) %>%
  arrange(rank) %>%
  select(-rank) %>%
  kable(format = 'latex', col.names = c('Category','Class','Sample Size','Training','Validation','Training','Validation','Training','Validation')) %>%
  add_header_above(c(' '=3, 'Sensitivity' = 2, 'Specificity' = 2, 'F1 Score' = 2))



#---------------
# Below here is exploratory stuff
#---------------


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
