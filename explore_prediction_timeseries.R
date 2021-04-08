library(tidyverse)
library(patchwork)
library(ggnewscale)
library(zoo)

source('config.R')

predictions = read_csv('./data/vgg16_v1_60epochs_predictions.csv') %>%
  mutate(file = str_remove(file, '.jpg')) %>%
  separate(file, c('phenocam_name','datetime'), '_', extra='merge') %>%
  mutate(datetime = str_replace_all(datetime,'_','-')) %>%
  mutate(datetime = lubridate::as_datetime(datetime, format='%Y-%m-%d-%H%M%S')) %>%
  mutate(hour = lubridate::hour(datetime),
         date = lubridate::date(datetime)) %>%
  select(-filepath,-hour,-datetime) %>%
  arrange(date)

ingest_transition_file = function(f){
  read_csv(f, skip = 16) %>%
    filter(gcc_value == 'gcc_90') %>%
    select(phenocam_name = site, direction, transition_10, transition_25, transition_50) %>%
    gather(threshold, date, -direction, -phenocam_name) %>%
    arrange(date) %>%
    mutate(original_transition_date = TRUE)
}

all_gcc_files = list.files(phenocam_gcc_folder, pattern = '1day_transition_dates.csv', full.names = TRUE)

all_site_transitions = map_df(all_gcc_files, ingest_transition_file)
#--------------------------------


three_category_figure = function(phenocam, daily_max_window=10, smoothing_window = 10,
                                 start_date = NA, end_date = NA){

  site_probability = predictions %>%
    filter(phenocam_name == phenocam) %>%
    pivot_longer(c(-phenocam_name,-date), names_sep='-', names_to=c('category','class'), values_to='probability') %>%
    group_by(phenocam_name, category, class) %>%
    mutate(probability = rollmean(probability, k=smoothing_window, fill=NA)) %>%
    ungroup()
  site_transitions = all_site_transitions %>%
    filter(phenocam_name == phenocam)
  
  if(!is.na(start_date)){
    site_probability = site_probability %>%
      filter(date >= start_date, date <= end_date)
    site_transitions = site_transitions %>%
      filter(date >= start_date, date <= end_date)
  }
  
  
  daily_max = site_probability %>%
    group_by(phenocam_name, category, date) %>%
    summarize(daily_max_probability = max(probability)) %>%
    ungroup() %>%
    group_by(phenocam_name, category) %>%
    mutate(daily_max_probability = rollmean(daily_max_probability, daily_max_window, fill=NA)) %>%
    ungroup() %>%
    as_tibble()
  
  figs = list()
  i = 1
  for(this_category in c('dominant_cover','crop_type','crop_status')){
    
    plot_title = ifelse(i==1, phenocam, '')
    show_transition_legend = ifelse(i==1, T, F)
    figs[[i]] = ggplot(filter(site_probability, category==this_category), aes(x=date, color=fct_rev(class))) + 
      geom_line(aes(y=probability),size=1, linetype='solid') + 
      scale_color_brewer(palette = 'Dark2') + 
      geom_line(data=filter(daily_max, category==this_category), aes(x=date,y=daily_max_probability),size=1, color='black',linetype='dotted', inherit.aes = F) + 
      ggnewscale::new_scale_color() +
      geom_vline(data=site_transitions, aes(xintercept=date,color=fct_rev(paste(direction,threshold))), size=1, show.legend=show_transition_legend) +
      scale_color_viridis_d(direction = -1) + 
      scale_x_date(date_breaks = '1 month') +
      scale_y_continuous(limits=c(0,1.2)) + 
      facet_wrap(~category, ncol=1) +
      ggtitle(plot_title) +
      theme_bw(20) + 
      theme(legend.position = 'bottom')
    
    i = i+1
  }
  
  wrap_plots(figs) + plot_layout(ncol=1, guides = 'auto')
  
}

prediction_site_years = predictions %>%
  mutate(year = lubridate::year(date)) %>%
  group_by(phenocam_name, year) %>%
  summarise(start_date = min(date),
            end_date   = max(date)) %>%
  ungroup() %>%
  mutate(duration = end_date - start_date) %>%
  filter(duration > 60)


for(i in 1:nrow(prediction_site_years)){
#for(i in sample(1:200,5)){
  fig = three_category_figure(phenocam = prediction_site_years$phenocam_name[i], 
                              daily_max_window = 10,
                              start_date = prediction_site_years$start_date[i], 
                              end_date = prediction_site_years$end_date[i])
  
  figure_folder = './prediction_timeseries_figures/'
  figure_filename = paste0(prediction_site_years$phenocam_name[i],'-',prediction_site_years$year[i],'.png')
  ggsave(paste0(figure_folder, figure_filename), fig, width=60, height=50, units='cm', dpi=100)
}
