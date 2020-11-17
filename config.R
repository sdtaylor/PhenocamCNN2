

# ------------------
# for initial processing/downloading of training images
# ------------------
phenocam_site_file = 'site_list.csv'
phenocam_gcc_folder = 'data/phenocam_gcc/'
phenocam_training_image_folder = 'data/phenocam_train_images/'

# the number of images sampled per site/year/period
# for initial training data gathering
random_images_per_period = 50

training_images_download_list = 'images_for_download.csv'

pasture_cameras = c("archboldavir","archboldavirx","archboldbahia","archboldpnot","archboldpnotx",'tworfpr',"harvardfarmnorth","harvardfarmsouth","harvardgarden",
                    'rosemountnprs','sweetbriargrass','meadpasture','NEON.D04.LAJA.DP1.00033','wolfesneckfarm')