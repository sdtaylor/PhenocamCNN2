import numpy as np
from keras_preprocessing.image import (load_img,
                                       img_to_array,
                                       )

from skimage.io import imread
from skimage.transform import resize

from warnings import warn

# Load in images to numpy arrays
def load_imgs_from_df(df, x_col, img_dir, target_size, data_format='channels_last'):
    """
    From a dataframe load all images into a numpy array with final shape
    (n_images, height, width, 3), where height and width are specified in 
    target_shape.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe containing image info.
    x_col : str
        column in df which contains the image filenames.
    img_dir : str
        folder containing all the images.
    target_size : tuple
        final size to transform images to
    data_format : TYPE, optional
        DESCRIPTION. The default is 'channels_last'.

    Returns
    -------
    img_array : np.array
        array of images

    """
    
    n_images = len(df)
    img_array = np.zeros((n_images,) + target_size + (3,), dtype=np.int32)
    
    for i,j in enumerate(df[x_col]):
        img = load_img(img_dir + j,
                       color_mode='rgb',
                       target_size=target_size)
        img_array[i] = img_to_array(img, data_format=data_format)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
    
    return img_array


def random_crop(img, crop_dim, height_cutoff=0.5):
    """
    Pull a crop out of an image of size (crop_dim,crop_dim)
    Height cutoff is the percentage below which the crop will be made.
    ie. 0.8 means the crop comes from the bottom 80% of the image
    """
    assert height_cutoff <= 1 and height_cutoff > 0, 'height cutoff must be between 0-1'
    assert crop_dim < img.shape[0] and crop_dim < img.shape[1], 'crop_dim must be < img dims'
    # Inverse this actual subset starts counting at the top of the image
    height_cutoff = 1 - height_cutoff
    
    img_height = img.shape[0]
    max_crop_height = int(img_height * height_cutoff)

    crop_dim_too_large = img_height - max_crop_height < crop_dim
    if crop_dim_too_large:
        warn('crop + height cutoff is too small, increasing height_cutoff')
        while crop_dim_too_large:
            height_cutoff -= 0.1
            max_crop_height = int(img_height * height_cutoff)
            crop_dim_too_large = img_height - max_crop_height < crop_dim
    
    img = img[max_crop_height:]
    
    axis0_start = np.random.randint(1, img.shape[0] - crop_dim)
    axis1_start = np.random.randint(1, img.shape[1] - crop_dim)
    
    return img[axis0_start:(axis0_start+crop_dim), axis1_start:(axis1_start+crop_dim)].copy()


def generate_random_crops(df, x_col, y_col, 
                          n_random_crops_per_image, crop_dim, crop_height,
                          img_dir, final_target_size, data_format='channels_last'):
    
    final_sample_size = len(df) * n_random_crops_per_image
    
    img_array = np.zeros((final_sample_size,) + final_target_size + (3,), dtype=np.uint8)
    
    # if y_col is a single column vs multiple columns for multiple binary categories
    if isinstance(y_col, str):
        y_array = np.zeros(final_sample_size, dtype=np.int32)
    elif isinstance(y_col, list):
        assert all([isinstance(y,str) for y in y_col]), 'y_col must be list of strs'
        y_array = np.zeros((final_sample_size,len(y_col)), dtype=np.int32)
    else:
        ValueError('y_col must be str or list of strs matching target column or columns in df')
        
        
    i=0
    for index, row in df.iterrows():
        img = imread(img_dir + row[x_col])
        
        for random_i in range(n_random_crops_per_image):
            cropped = random_crop(img, crop_dim, height_cutoff=crop_height)
            img_array[i] = resize(cropped, final_target_size, preserve_range=True).astype(np.uint8)
            y_array[i]   = row[y_col].values
            i+=1
    
    return img_array, y_array
        

