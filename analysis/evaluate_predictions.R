library(tidyverse)
library(zoo)
#library(kableExtra)

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

# original_training_data %>%
#   pivot_longer(c('crop_status','dominant_cover','crop_type'), names_to='category', values_to='class_value') %>%
#   left_join(class_descriptions, by=c('category','class_value')) %>%
#   select(phenocam_name, date, category, class = class_description) %>%
#   pivot_wider(names_from='category', values_from='class') %>%
#   count(dominant_cover, crop_type, crop_status) %>%
#   View()



# Load the extra images and annotate using the hand annotated data
# extra_image_training_data = read_csv('data/extra_images_for_fitting.csv') %>%
#   select(phenocam_name, date, filename) %>%
#   left_join(select(original_training_data,-filename), by=c('phenocam_name','date')) %>%
#   filter(complete.cases(.)) # some extra images which don't have a date in the original_training_data may have snuck in

all_training_data = original_training_data %>%
  #bind_rows(extra_image_training_data) %>% 
  inner_join(val_train_split_info, by='filename') %>%
  distinct() %>%   # this distinct drops any duplicates, inevitable with 100k+ images
  pivot_longer(c('crop_status','dominant_cover','crop_type'), names_to='category', values_to='class_value') %>%
  left_join(class_descriptions, by=c('category','class_value')) %>%
  select(phenocam_name, filename, year, date, category, data_type, true_class = class_description)

# some numbers for the manuscript
validation_sites = c('arsmorris2', 'mandani2', 'cafboydnorthltar01')

val_train_split_info %>%
  separate(filename, into=c('phenocam_name'), remove = F, extra='drop') %>%
  mutate(is_validation_site = phenocam_name %in% c('arsmorris2', 'mandani2', 'cafboydnorthltar01')) %>%
  count(data_type, is_validation_site)




n_validation_site_images = all_training_data %>%
  filter(phenocam_name %in% validation_sites) %>%
  filter(data_type=='val') %>%
  distinct(phenocam_name, filename) %>%
  nrow()

all_training_data %>%
  filter(data_type=='val') %>%
  mutate(is_validation_site = phenocam_name %in% c('arsmorris2', 'mandani2', 'cafboydnorthltar01')) %>%
  group_by(is_validation_site) %>%
  summarise(n_images = n_distinct(filename)) %>%
  ungroup()


sample_sizes = all_training_data %>%
  count(category, true_class, data_type) %>%
  rename(class = true_class, sample_size=n)

# order things correctly for the figures. These are in reverse since ggplot starts
# from the bottom
class_order = c('blurry','no_crop','water','snow','residue','soil','vegetation',
                'unknown_plant','other','alfalfa','soybean','wheat/barley','corn',
                'senesced','senescing','flowers','growth','emergence',
                'overall')
class_labels = str_to_title(str_replace(class_order,'_',' '))

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

#-------------------------
# Evaluation of the original VGG16 predictions before post-processing
#-------------------------

# sanity check that all training data used here for validation actually has a max probability class assigned.
all_training_data %>%
  left_join(select(max_probability,filename,category,maxprob_class), by=c('filename','category')) %>%
  group_by(filename) %>% 
  summarise(has_maxprob_class = sum(is.na(maxprob_class))==0) %>%
  ungroup() %>%
  summarise(percent_with_maxprob=mean(has_maxprob_class),
            num_with_maxprob=sum(has_maxprob_class))

original_classification_data = all_training_data %>%
  left_join(select(max_probability,filename,category,maxprob_class), by=c('filename','category'))

original_prediction_stats = original_classification_data %>%
  rename(predicted_class = maxprob_class) %>%
  group_by(category, data_type) %>%
  nest() %>%
  mutate(stats = map(data, ~get_stats(., stats=c('precision','recall','f1')))) %>%
  select(-data) %>%
  unnest(stats) %>%
  left_join(sample_sizes, by=c('category','class','data_type'))


# summary stats for each category
original_prediction_overal_stats = original_prediction_stats %>%
  group_by(category, term, data_type) %>%
  summarise(estimate = sum(estimate*sample_size)/sum(sample_size),
            sample_size=sum(sample_size)) %>%
  mutate(class = 'overall')

# TODO: maybe put the crop_status in the correct biological order, 
original_prediction_stats = original_prediction_stats %>%
    bind_rows(original_prediction_overal_stats) %>%
    mutate(class = factor(class, levels = class_order, labels=class_labels, ordered = T)) %>%
    mutate(facet_label = paste0(str_to_title(term),': ',str_to_title(str_replace(category,'_',' ')))) 

original_prediction_error_labels = original_prediction_stats %>%
    mutate(estimate = trimws(format(round(estimate,2),drop0trailing=F))) %>%
    pivot_wider(names_from='data_type', values_from=c('estimate','sample_size')) %>%
    mutate(error_text = paste0(estimate_val,', ',estimate_train,' (',sample_size_val+sample_size_train,')'))
    
base_stats_figure = original_prediction_stats %>%
    select(-sample_size) %>%
    pivot_wider(names_from='data_type', values_from='estimate') %>%
ggplot(aes(y=class)) + 
    geom_segment(aes(x=0,xend=train,yend=class), color='grey60', size=8) + 
    geom_segment(aes(x=0,xend=val,yend=class), size=3) + 
    geom_text(data=original_prediction_error_labels, aes(x=1.025, label=error_text), size=4, hjust=0) + 
    scale_x_continuous(breaks=c(0.3,0.5,0.7,0.9), labels=c('0.3','0.5','0.7','0.9')) + 
    coord_cartesian(xlim=c(0.25,1.8)) + 
    facet_wrap(~fct_rev(facet_label), ncol=3, scales='free_y') +
    theme_bw() +
    theme(axis.title.y = element_blank(),
          axis.title.x = element_text(size=14),
          axis.text.y  = element_text(color='black', size=13.5),
          axis.text.x  = element_text(color='black', size=12),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(color='grey80'),
          panel.grid.minor   = element_blank(),
          strip.background = element_blank(),
          strip.text = element_text(size=14, hjust=0)) +
    labs(x='Accuracy Metric Value')

ggsave('./manuscript/figures/fig2_base_stats.pdf', plot=base_stats_figure, height=32, width=28, unit='cm', dpi=150)


# some confusion matrices
original_classification_data %>%
  count(category, data_type, true_class, maxprob_class) %>%
  group_by(category) %>%
  complete(data_type, true_class, maxprob_class, fill=list(n=0)) %>%
  ungroup() %>% 
  ggplot(aes(x=str_wrap(true_class,8), y=maxprob_class)) +
  geom_tile(size=0.5, color='black', alpha=0) + 
  geom_text(aes(label=n),size=5) +
  facet_wrap(category~data_type, scales='free',ncol=2) +
  theme_minimal(10) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(color='black'),
        axis.title = element_text(size=20),
        strip.text = element_text(size=18),
        axis.text = element_text(size=12, color='black'),
        axis.text.x = element_text(angle=-45, hjust=0)) +
  labs(x='True Class', y='Predicted class from VGG16 model')


#-------------------------
# Evaluation of the predictions after post-processing (ie. after the HMM step)
#-------------------------

#---------------------------------
# sanity check that all training data used here for validation actually has an hmm class assigned.
# some images will be lost due to incomplete/short timeseries.
all_training_data %>%
  left_join(hmm_predictions, by=c('phenocam_name','date','category')) %>%
  group_by(phenocam_name, year.x, filename) %>% 
  summarise(has_hmm_class = sum(is.na(hmm_class))==0) %>%
  ungroup() %>%
  summarise(percent_with_hmm=mean(has_hmm_class),
          num_with_hmm=sum(has_hmm_class))

# Remove blurry since it's not in the data post-processing
# Remove unknown plant class. Crop type category is treated
# in a special way in the post-processing, so performance metrics 
# of unknown plant class woudl be uninformative.
hmm_classification_data = all_training_data %>%
  left_join(hmm_predictions, by=c('phenocam_name','date','category')) %>%
  #filter(category != 'crop_type') %>%
  filter(true_class != 'blurry') %>%
  filter(true_class != 'unknown_plant') %>%
  filter(hmm_class != 'unknown_plant') 

hmm_prediction_stats = hmm_classification_data %>%
  rename(predicted_class = hmm_class) %>%
  group_by(category, data_type) %>%
  nest() %>%
  mutate(stats = map(data, ~get_stats(., stats=c('precision','recall','f1')))) %>%
  select(-data) %>%
  unnest(stats) %>%
  left_join(sample_sizes, by=c('category','class','data_type'))

hmm_overal_stats = hmm_prediction_stats %>%
  group_by(category, term, data_type) %>%
  summarise(estimate = sum(estimate*sample_size)/sum(sample_size),
            sample_size=sum(sample_size)) %>%
  mutate(class = 'overall')

hmm_prediction_stats = hmm_prediction_stats %>%
  bind_rows(hmm_overal_stats) %>%
  mutate(class = factor(class, levels = class_order, labels=class_labels, ordered = T)) %>%
  mutate(facet_label = paste0(str_to_title(term),': ',str_to_title(str_replace(category,'_',' ')))) 

hmm_error_labels = hmm_prediction_stats %>%
  mutate(estimate = trimws(format(round(estimate,2),drop0trailing=F))) %>%
  pivot_wider(names_from='data_type', values_from=c('estimate','sample_size')) %>%
  mutate(error_text = paste0(estimate_val,', ',estimate_train,' (',sample_size_val+sample_size_train,')'))

hmm_stat_figure = hmm_prediction_stats %>%
  select(-sample_size) %>%
  pivot_wider(names_from='data_type', values_from='estimate') %>%
  ggplot(aes(y=class)) + 
  geom_segment(aes(x=0,xend=train,yend=class), color='grey60', size=8) + 
  geom_segment(aes(x=0,xend=val,yend=class), size=3) + 
  geom_text(data=hmm_error_labels, aes(x=1.025, label=error_text), size=4, hjust=0) + 
  scale_x_continuous(breaks=c(0.3,0.5,0.7,0.9), labels=c('0.3','0.5','0.7','0.9')) + 
  coord_cartesian(xlim=c(0.25,1.8)) + 
  facet_wrap(~fct_rev(facet_label), ncol=3, scales='free_y') +
  theme_bw() +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_text(size=14),
        axis.text.y  = element_text(color='black', size=13.5),
        axis.text.x  = element_text(color='black', size=12),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color='grey80'),
        panel.grid.minor   = element_blank(),
        strip.background = element_blank(),
        strip.text = element_text(size=14, hjust=0)) +
  labs(x='Accuracy Metric Value')

ggsave('./manuscript/figures/fig3_hmm_stats.pdf', plot=hmm_stat_figure, height=28, width=28, unit='cm', dpi=150)




hmm_classification_data %>%
  count(category, data_type, true_class, hmm_class) %>%
  group_by(category) %>%
  complete(data_type, true_class, hmm_class, fill=list(n=0)) %>%
  ungroup() %>% 
  ggplot(aes(x=str_wrap(true_class,8), y=hmm_class)) +
  geom_tile(size=0.5, color='black', alpha=0) + 
  geom_text(aes(label=n),size=5) +
  facet_wrap(category~data_type, scales='free',ncol=2) +
  theme_minimal(10) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(color='black'),
        axis.title = element_text(size=20),
        strip.text = element_text(size=18),
        axis.text = element_text(size=12, color='black'),
        axis.text.x = element_text(angle=-45, hjust=0)) +
  labs(x='True Class', y='Predicted class after post-processing')
