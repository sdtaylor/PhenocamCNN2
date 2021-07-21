library(tidyverse)
library(patchwork)
library(ggnewscale)
#library(zoo)

source('classification_postprocessing/postprocessing_tools.R')

#-----------------
# like explore_prediction_timeseries.R but trying some slightly
# different summarization.

arrange_crop_status = function(df){
  class_order = c('blurry','no_crop','water','snow','residue','soil','vegetation',
                  'unknown_plant','other','alfalfa','soybean','wheat/barley','corn',
                  'senesced','senescing','flowers','growth','emergence',
                  'overall','MaxP','HMM')
  df %>%
    mutate(class = factor(class, levels = class_order,ordered = T))
}

hmm_predictions = read_csv('./data/final_predictions.csv') %>%
  pivot_longer(c('dominant_cover','crop_status','crop_type'), names_to='category', values_to='class')

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
  select(phenocam_name, filename, year, date, category, class)


phenocam_to_plot = 'arsope3ltar'
year_to_plot     = 2019

  
  figs = list()
  i = 1
  
  for(this_category in c('dominant_cover','crop_status','crop_type')){
  #for(this_category in c('crop_type','dominant_cover','crop_status')){
    site_predictions = original_predictions %>%
      filter(phenocam_name == phenocam_to_plot, year==year_to_plot) %>%
      filter(category == this_category) %>%
      arrange_crop_status() %>%
      as_tibble()
    
    site_max_prob = max_probability %>%
      filter(phenocam_name == phenocam_to_plot, year==year_to_plot) %>%
      filter(category == this_category) %>%
      arrange_crop_status() %>%
      as_tibble()
    
    site_hmm = hmm_predictions %>%
      filter(phenocam_name == phenocam_to_plot, year==year_to_plot) %>%
      filter(category == this_category)%>%
      arrange_crop_status() %>%
      as_tibble()
    
    
      
    if(this_category == 'crop_status') {
      plot_subtitle = 'Crop Status'
      #color_palette = c('red','#d55e00', viridis::viridis(5, end=0.85, direction = -1))
      color_palette = c('#cc79a7','#d55e00','#582f0e','#a68a64','#007f5f','#80b918','#d4d700')
    } else if(this_category == 'dominant_cover') {
      plot_subtitle = 'Dominant Cover'
      color_palette = c('#cc79a7','#0072b2','grey60','#d55e00','#e69f00','#80b918')
    } else if(this_category == 'crop_type') {
      plot_subtitle = 'Crop Type'
      color_palette = c("#cc79a7", "#d55e00", "#000000", "#E69F00", "#56B4E9", "#009E73",'#66a61e', "#F0E442")
    }
  
    # for the given category get all th relevant classes, in the correct order,
    # and include entires for the MaxP and HMM stuff on top
    category_classes = levels(site_predictions$class)                                              # extract from all levels first to get correct order
    category_classes = category_classes[category_classes %in% unique(site_predictions$class)]      # only classes for this category          
    
    # nice labels for plot
    class_labels    = str_to_title(str_replace(category_classes,'_',' '))
    
    # the two top categories
    category_classes = c(category_classes, 'MaxP','HMM')  
    class_labels     = c(class_labels, 'MaxP','HMM')
    
    # MaxP and HMM geom_segment size
    segment_height = 0.2
    segment_size = 0.8
    
    # alternative style plot with probabilites as lines like a timeseries.
    # figs[[i]] =  ggplot(site_predictions, aes(x=date, y=probability)) +
    #   geom_path(aes(color=class), size=0.75) +
    #   geom_hline(yintercept = 1.2, size=8, color='grey80') +      # the grey background behind HMM and MaxP lines
    #   geom_hline(yintercept = 1.4, size=8, color='grey80') + 
    #   geom_segment(data=site_max_prob, aes(y=1.15,yend=1.25,x=date,xend=date,color=class),size=0.8) + # HMM and MaP bars, each date is a vertical
    #   geom_segment(data=site_hmm,      aes(y=1.35,yend=1.45,x=date,xend=date,color=class),size=0.8) + # segment with respective color
    #   scale_color_manual(values = color_palette) +
    #   #scale_x_date(limits = as.Date(c('2019-07-15','2019-08-20')), date_breaks = '4 day', date_labels = '%b. %d') + 
    #   scale_x_date(date_breaks = '2 month', date_labels = '%b. %d', expand = expansion(0.01)) + 
    #   scale_y_continuous(breaks = c(0,0.25,0.5,0.75,1.0,1.2,1.4), labels=c('0','0.25','0.50','0.75','1.0','MaxP','HMM')) + 
    #   theme_bw() +
    #   theme(axis.text = element_text(color='black', size=10),
    #         axis.title.y = element_text(size=12),
    #         legend.text = element_text(size=10),
    #         legend.position = 'right',
    #         plot.subtitle = element_text(size=14),
    #         panel.background = element_rect(fill='grey95')) + 
    #   labs(y='Probability',x='',color='', subtitle = plot_subtitle) +
    #   guides(color=guide_legend(ncol=2, override.aes = list(size=6))) +
    #   theme()
    
    # with probabilites as circles
    f = ggplot(site_predictions, aes(x=date, y=class)) +
      geom_point(aes(color=class, size=probability)) +
      scale_color_manual(values = color_palette) +
      scale_x_date(date_breaks = '2 month', date_labels = '%b. %d', expand = expansion(0.01)) +
      scale_y_discrete(limits=category_classes, labels = class_labels) +
      theme_bw() +
      theme(axis.text = element_text(color='black', size=10),
            axis.title.y = element_blank(),
            legend.text = element_text(size=10),
            legend.title = element_text(size=12),
            legend.position = ifelse(i==3, 'bottom','none'),
            plot.subtitle = element_text(size=14),
            panel.background = element_rect(fill='grey95')) +
      labs(size='Probability',x='',color='', subtitle = plot_subtitle) +
      guides(color=guide_none(),
             size=guide_legend(label.position = 'top')) +
      theme()

    # doing these separately for each category because it's a *very* funky plot.
    if(this_category == 'dominant_cover') {
      figs[[i]] = f +
        geom_hline(yintercept = 7, size=8, color='grey80') +      # the grey background behind HMM and MaxP lines
        geom_hline(yintercept = 8, size=8, color='grey80') + 
        geom_segment(data=site_max_prob, aes(y=7-segment_height,yend=7+segment_height,x=date,xend=date,color=class),size=segment_size) + # HMM and MaP bars, each date is a vertical
        geom_segment(data=site_hmm,      aes(y=8-segment_height,yend=8+segment_height,x=date,xend=date,color=class),size=segment_size)   # segment with respective color
    } else if(this_category == 'crop_status') {
      figs[[i]] = f +
        geom_hline(yintercept = 8, size=8, color='grey80') +      # the grey background behind HMM and MaxP lines
        geom_hline(yintercept = 9, size=8, color='grey80') + 
        geom_segment(data=site_max_prob, aes(y=8-segment_height,yend=8+segment_height,x=date,xend=date,color=class),size=segment_size) + # HMM and MaP bars, each date is a vertical
        geom_segment(data=site_hmm,      aes(y=9-segment_height,yend=9+segment_height,x=date,xend=date,color=class),size=segment_size)   # segment with respective color
    } else if(this_category == 'crop_type') {
      figs[[i]] = f +
        geom_hline(yintercept = 9, size=8, color='grey80') +      # the grey background behind HMM and MaxP lines
        geom_hline(yintercept = 10, size=8, color='grey80') + 
        geom_segment(data=site_max_prob, aes(y=9-segment_height,yend=9+segment_height,x=date,xend=date,color=class),size=segment_size) + # HMM and MaP bars, each date is a vertical
        geom_segment(data=site_hmm,      aes(y=10-segment_height,yend=10+segment_height,x=date,xend=date,color=class),size=segment_size)   # segment with respective color
    }
      
    i=i+1
  
  }
  
  wrap_plots(figs) + plot_layout(ncol=1)
  
               