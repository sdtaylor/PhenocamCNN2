library(tidyverse)
library(rnaturalearth)
library(sf)
library(kableExtra)

basemap = rnaturalearth::ne_states(country='United States of America',returnclass = 'sf') %>%
  filter(!name %in% c('Alaska','Hawaii'))

ltar_sites = phenocamr::list_sites() %>%
  mutate(is_ltar = str_detect(group, 'LTAR')) %>%
  filter(is_ltar) %>%
  pull(site)

sites = read_csv('site_list.csv') %>%
  st_as_sf(coords = c('lon','lat'), crs=4326) %>%
  mutate(is_ltar = phenocam_name %in% ltar_sites)

ggplot() + 
  geom_sf(data=basemap) +
  geom_sf(data=filter(sites,!is_ltar), size=6) +
  geom_sf(data=filter(sites,is_ltar), aes(color='LTAR'), shape=17, size=6) +
  scale_color_manual(values=c('red')) + 
  coord_sf(xlim=c(-125,-68)) +
  theme_bw(25) +
  theme(legend.position = c(0.85,0.3),
        legend.title = element_blank(),
        legend.background = element_rect(fill='white', color='black'))


#---------------------
# And a table
read_csv('site_list.csv') %>%
  mutate(ltar_site = ifelse(phenocam_name %in% ltar_sites, 'yes','no')) %>%
  mutate(number=row_number()) %>%
  select(number, phenocam_name, ltar_site, everything()) %>%
  kable(format='simple')
  