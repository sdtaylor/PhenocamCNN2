import numpy as np
from keras_preprocessing.image import ImageDataGenerator

class MultiOutputDataGenerator(ImageDataGenerator):
    """
    A generator for multiple output models. From 
    https://github.com/keras-team/keras/issues/12639#issuecomment-506338552
    """
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict
            
            
def keras_predict(df, filename_col, model, target_size, preprocess_func,
                  image_dir=None, predict_prob=True, chunksize=500):
    """
    Load a keras model and predict on all images specified in the filename_col,
    of df

    """
    df = df.copy()
    df['class'] = 'a' # need a dummy class column to pass to the generator
    
    g  = ImageDataGenerator(preprocessing_function=preprocess_func).flow_from_dataframe(
                                         df, 
                                         directory = image_dir,
                                         target_size = target_size,
                                         batch_size = chunksize,
                                         shuffle = False,
                                         x_col = filename_col,
                                         y_col = 'class'
                                         )
    
    predictions = model.predict(g, workers=32, verbose=1)
    
    if predict_prob:
        return predictions
    else:
        return np.argmax(predictions, 1)