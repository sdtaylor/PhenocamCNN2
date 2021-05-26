

# ------------------
# for initial processing/downloading of training images
# ------------------
phenocam_site_file = 'site_list.csv'
phenocam_gcc_folder = 'data/phenocam_gcc/'
phenocam_training_image_folder = 'data/phenocam_train_images/'

# 
extra_training_image_folder = 'data/extra_phenocam_train_images/'

# the number of images sampled per site/year/period
# for initial training data gathering
random_images_per_period = 50

training_images_download_list = 'images_for_download.csv'

# ag sites which are actually pastures
.pasture_cameras  = c("uiefprairie2", 
                      "archboldavir", 
                      "archboldavirx", 
                      "archboldbahia", 
                      "archboldpnot", 
                      "archboldpnotx", 
                      "tworfpr", 
                      "harvardfarmnorth", 
                      "harvardfarmsouth", 
                      "harvardgarden", 
                      "rosemountnprs", 
                      "sweetbriargrass", 
                      "kelloggrestoredprairie",
                      "meadpasture",
                      "meadpasturesw",
                      "meadpasturese",
                      "NEON.D04.LAJA.DP1.00033", 
                      "wolfesneckfarm")

# ag sites with 1-several experimental plots
.experimental_cameras = c('arsmnswanlake1',
                         'kelloggcorn2',
                         'kelloggcorn3',
                         'kelloggmiscanthus',
                         'kelloggoldfield',
                         'kingmanfarm',
                         'NEON.D10.STER',
                         'silverton',
                         'tworfta')

# ones to exclude for other reasons.
.others_to_exclude = c('turkeypointenf02',
                       'armoklahoma',
                      'tidmarshplymouth')

ag_sites_to_exclude = c(.pasture_cameras, .experimental_cameras, .others_to_exclude)