library(tidyverse)

#########################################################################
# Take doy of the current season and return Mar. 1, Jan. 30, etc.
doy_to_date = function(x){
  dates = as.Date(paste('2018', x,sep='-'), '%Y-%j')
  abbr  = strftime(dates, '%b %d')
  return(abbr)
}

#-----------------------
site_gcc = read_csv('./data/phenocam_gcc/arsmorris1_AG_1000_1day.csv', skip = 24) %>%
  filter(year==2018) %>%
  select(date, year, doy, gcc =smooth_gcc_90)

final_results = read_csv('./data/final_predictions.csv') %>%
  filter(phenocam_name=='arsmorris1', year==2018) %>%
  mutate(doy = lubridate::yday(date))

#-------------------------

#                   "emergence" "flowers"   "growth"    "no_crop"   "senesced"  "senescing"
crop_status_colors = c('#d4d700', '#007f5f','#80b918', '#d55e00', '#582f0e','#a68a64')

ybase = 0.45
seg_height = 0.006
seg_size=1

figure_abstract_base = ggplot(site_gcc, aes(x=doy)) + 
  geom_point(aes(y=gcc, color=gcc), size=6) + 
  scale_color_gradient(low='darkorange4', high='green1') +
  ggnewscale::new_scale_color() + 
  geom_segment(data=final_results, aes(y=ybase-seg_height,yend=ybase+seg_height,x=doy,xend=doy,color=crop_status),size=seg_size) +
  scale_color_manual(values=crop_status_colors) +
  scale_x_continuous(limits = c(-10,365), breaks = c(15,74,135,196,258,319), labels = doy_to_date) + 
  theme_minimal() +
  theme(legend.position = 'none',
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(size=24, color='black'),
        axis.ticks = element_blank())+
  labs(x='',y='')

ggsave('manuscript/figures/graphical_abstract_base.png',figure_abstract_base, width=11, height=6, dpi=300)  
  
