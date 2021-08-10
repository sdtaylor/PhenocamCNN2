library(tidyverse)
library(magick)
library(cowplot)

source('classification_postprocessing/postprocessing_tools.R')
source('config.R')



# order things correctly for the figures. These are in reverse since ggplot starts
# from the bottom
class_order = c('blurry','no_crop','water','snow','residue','soil','vegetation',
                'unknown_plant','other','alfalfa','soybean','wheat/barley','corn',
                'senesced','senescing','flowers','growth','emergence',
                'overall')
class_labels = str_to_title(str_replace(class_order,'_',' '))

#-------------------------
# Prep model output ----

hmm_predictions = read_csv('./data/final_predictions.csv') %>%
  pivot_longer(c('dominant_cover','crop_status','crop_type'), names_to='category', values_to='hmm_class')

original_predictions = read_csv('./data/vgg16_v4_20epochs_predictions.csv') %>%
  mutate(filename=file) %>%
  prep_prediction_dataframe() %>% 
  prediction_df_wide_to_long()
# 
max_probability = original_predictions %>%
  group_by(filename, category, class) %>% # deal with >1 photo per day
  summarise(probability = mean(probability), n=n()) %>%
  ungroup() %>%
  group_by(filename, category) %>%
  mutate(maxprob_class = class[which.max(probability)])
  ungroup() 

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

all_training_data = original_training_data %>%
  inner_join(val_train_split_info, by='filename') %>%
  distinct() %>%   # this distinct drops any duplicates, inevitable with 100k+ images
  pivot_longer(c('crop_status','dominant_cover','crop_type'), names_to='category', values_to='class_value') %>%
  left_join(class_descriptions, by=c('category','class_value')) %>%
  select(phenocam_name, filename, year, date, category, data_type, true_class = class_description)

original_classification_data = all_training_data %>%
  left_join(select(max_probability,filename,category,maxprob_class), by=c('filename','category'))

#---------------------------------

make_diagnostic_plot = function(image_of_interest){

  #image_of_interest = 'arsmorris2_2018_11_06_120000.jpg'
  image_path = paste0(phenocam_training_image_folder,image_of_interest)
  
  image_classification = all_training_data %>%
    filter(filename==image_of_interest) %>%
    select(phenocam_name, date, filename, category, true_class, data_type)
  
  image_predictions = max_probability %>%
    filter(filename==image_of_interest) %>%
    left_join(image_classification, by=c('filename','category')) %>%
    group_by(filename, category) %>%
    mutate(is_maxprob_class = class==maxprob_class,
           is_true_class = class==true_class) %>%
    ungroup() %>%
    mutate(facet_label = str_to_title(str_replace(category,'_',' '))) %>%
    mutate(class = factor(class, levels = class_order, labels=class_labels, ordered = T)) %>%
    ##filter(class==maxprob_class) %>%
    #mutate(maxprob_class = paste0(maxprob_class,' (',round(probability,2),')')) %>%
    #select(filename, category, maxprob_class) %>%
    as_tibble()
  
  true_class_label = image_predictions %>%
    filter(is_true_class) %>%
    mutate(label_x = ifelse(probability<0.5, 0.6, 0.2))
  
  data_plot=ggplot(image_predictions, aes(x=probability,y=class)) + 
    geom_col(aes(fill=is_maxprob_class)) + 
    geom_label(data=true_class_label, aes(x=label_x), label='Human Annotation', size=1.5, label.padding = unit(0.1,'lines')) + 
    scale_fill_manual(values=c('black','#009e73')) + 
    scale_x_continuous(breaks=c(0,0.2,0.4,0.6,0.8,1.0)) + 
    coord_cartesian(xlim=c(0,1.0)) + 
    facet_wrap(~fct_rev(facet_label), ncol=1, scales='free_y') +
    theme_bw() +
    theme(axis.title.y = element_blank(),
          axis.title.x = element_text(size=7),
          axis.text  = element_text(color='black', size=6),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(color='grey80'),
          panel.grid.minor   = element_blank(),
          strip.background = element_blank(),
          strip.text = element_text(size=7, hjust=0),
          legend.position = 'none') +
    labs(x='Classification Probability')
  
  data_type = unique(image_classification$data_type)
  phenocam_name = unique(image_classification$phenocam_name)
  date = unique(image_classification$date)
  plot_title = paste(phenocam_name, date, data_type, sep=' - ')
  
  image_plot = ggdraw() + 
    draw_image(image_path, scale=0.98) +
    draw_text(plot_title,x=0.4,y=0.95, size=12)
  
  plot_grid(data_plot, image_plot, rel_widths = c(1, 2))

}


#img = 'arsmorris2_2018_11_06_120000.jpg'
random_training_images = sample(unique(all_training_data$filename), 50)

for(img in random_training_images){
  p = make_diagnostic_plot(img)
  img_filename = paste0('image_diagnostic_plots/diagnostic_',tools::file_path_sans_ext(img),'.png')
  cowplot::save_plot(filename = img_filename, plot=p, base_width = 6, base_height = 4, dpi=150)
}




