import numpy as np
import pandas as pd

from hmm_stuff.hmm_model_definitions import (dominant_cover_hmm_model,
                                             crop_type_hmm_model,
                                             crop_status_hmm_model,
                                             )




category_info = {'dominant_cover':{'prediction_file':'data/image_predictions_for_hmm-dominant_cover.csv',
                                   'observed_classes':['vegetation','residue','soil','snow','water'],
                                   'model_function':dominant_cover_hmm_model},
                 'crop_type':     {'prediction_file':'data/image_predictions_for_hmm-crop_type.csv',
                                   'observed_classes':['unknown_plant','large_grass','small_grass','other','fallow','no_crop'],
                                   'model_function':crop_type_hmm_model},
                 'crop_status':   {'prediction_file':'data/image_predictions_for_hmm-crop_status.csv',
                                   'observed_classes':['emergence','growth','flowers','senescing','senesced','no_crop'],
                                   'model_function':crop_status_hmm_model},
                 }

all_hmm_output = []

for category, category_attrs in category_info.items():
    pass

    image_predictions = pd.read_csv(category_attrs['prediction_file'])


    for p in image_predictions.phenocam_name.unique():
        site_subset = image_predictions[image_predictions.phenocam_name==p]
        
        for sequence in site_subset.site_sequence_id.unique():
            pass
        
            sequence_subset = site_subset[site_subset.site_sequence_id==sequence]
            
            nn_probabilites = sequence_subset[category_attrs['observed_classes']].values
            n_samples = nn_probabilites.shape[0]
            n_observed_classes = len(category_attrs['observed_classes'])
            
            # Load model from defintions file
            model = category_attrs['model_function'](nn_probabilites,
                                                     n_samples,
                                                     n_observed_classes)
            
           
            # model predictions and mapping to full hidden state names
            try:
                hmm_predictions_index = model.predict(list(range(n_samples)), algorithm='viterbi')
            except:
                print('HMM model failed for {} sequence_id: {}, category: {}'.format(p,sequence, category))
                continue
            
            hmm_predictions = [model.states[i].name for i in hmm_predictions_index]
            
            # setup a dataframe to store hmm results
            hmm_output = sequence_subset[['phenocam_name','year','date','site_sequence_id','category']].copy()           
            hmm_output['hmm_class'] =  hmm_predictions[1:] # drop the starting state
            
            all_hmm_output.append(hmm_output)

