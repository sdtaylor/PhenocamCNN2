import numpy as np

from pomegranate import (HiddenMarkovModel,
                         State,
                         DiscreteDistribution,
                         NeuralNetworkWrapper,
                         )

from json import dumps

"""
Here are the hidden markov model (HMM) descriptions. One for each
category (dominant_cover, crop_type, and crop_status).

Every unique phenocam sequence will technicially get it's own model since
the probabilites used as emessions for each are unique, even though
the states and transition probabilites are the same for each category.
"""

class NeuralNetworkWrapperCustom(NeuralNetworkWrapper):
    def __init__(self, predicted_probabilities, i, n_samples, n_classes):
        """
        A wrapper to use keras predicted probabilites directly as emmission 
        probabilites in the pomegranate HMM model. This replaces the DiscreteDistribution
        used in most of the pomegranate HMM examples.
        
        Derived from:
        https://github.com/jmschrei/pomegranate/blob/master/tutorials/C_Feature_Tutorial_6_Deep_Models.ipynb
        
        Parameters
        ----------
        predicted_probabilities : numpy 2d array
            predicted probabilites from keras with shape (timeseries_steps,n_classes).
        i : int
            which hidden state this wrapper represents. Must correspend with column
            i in the predicted_probabilites array
        n_samples : int
            timeseries length.
        n_classes : int
            number of classes.

        Returns
        -------
        None.

        """
        self.predicted_probabilities = predicted_probabilities
        self.i = i
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.d = 1
        
        assert self.predicted_probabilities.shape[0] == n_samples
        assert self.predicted_probabilities.shape[1] == n_classes
        
    def log_probability(self, timestep):
        i = int(timestep[0][0])
        assert i in range(0, self.n_samples)
        return np.log(self.predicted_probabilities[i,self.i:(self.i+1)])
    
    def from_summaries(self, inertia=0.0):
        pass
    
    def summarize(self, X, w):
        pass
    
    def clear_summaries(self):
        pass
    
    def to_json(self):
        """ 
        Required method for summary operations.
        """
        summary = {'NeuralNetworkWrapperCustom':{'class':self.i}}
        return dumps(summary)


def dominant_cover_hmm_model(nn_pobability_matrix,
                             timeseries_steps,
                             n_observed_classes):
    d0 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 0, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d1 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 1, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d2 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 2, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d3 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 3, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d4 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 4, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    
    s0_veg     = State(d0, name='vegetation')
    s1_residue = State(d1, name='residue')
    s2_soil    = State(d2, name='soil')
    s3_snow    = State(d3, name='snow')
    s4_water   = State(d4, name='water')
    
    model = HiddenMarkovModel()
    
    # Initialize each hidden state.
    # All states have an equal chance of being the starting state.
    for s in [s0_veg,s1_residue,s2_soil,s3_snow,s4_water]:
        model.add_state(s)
        model.add_transition(model.start, s, 1)
        
    model.add_transitions(s0_veg,     [s0_veg, s1_residue, s2_soil, s3_snow, s4_water],
                                      [95.,    1.25,       1.25,    1.25,    1.25])
    model.add_transitions(s1_residue, [s0_veg, s1_residue, s2_soil, s3_snow, s4_water],
                                      [1.25,   95.,        1.25,    1.25,    1.25])
    model.add_transitions(s2_soil,    [s0_veg, s1_residue, s2_soil, s3_snow, s4_water],
                                      [5/3.,   0.,         95.,     5/3.,    5/3.])
    model.add_transitions(s3_snow,    [s0_veg, s1_residue, s2_soil, s3_snow, s4_water],
                                      [1.25,   1.25,       1.25,    95.,     1.25])
    model.add_transitions(s4_water,   [s0_veg, s1_residue, s2_soil, s3_snow, s4_water],
                                      [1.25,   1.25,       1.25,    1.25,     95.])
        
    model.bake(verbose=False)
    
    return model

def crop_status_hmm_model(nn_pobability_matrix,
                          timeseries_steps,
                          n_observed_classes):
    # 0            1       2          3          4          5
    ['emergence','growth','flowers','senescing','senesced','no_crop']
    
    
    d0 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 0, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d1 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 1, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d2 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 2, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d3 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 3, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    d4 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 4, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)

    d5 = NeuralNetworkWrapperCustom(predicted_probabilities = nn_pobability_matrix, 
                                    i = 5, 
                                    n_samples = timeseries_steps, 
                                    n_classes = n_observed_classes)
    
    s0_emerge    = State(d0, name='emergence')
    s1_growth    = State(d1, name='growth')
    s2_fls       = State(d2, name='flowers')
    s3_sencing   = State(d3, name='senescing')
    s4_senced    = State(d4, name='senesced')
    s5_none      = State(d5, name='no_crop')
    
    model = HiddenMarkovModel()
    
    # Initialize each hidden state.
    # All states have an equal chance of being the starting state.
    for s in [s0_emerge,s1_growth,s2_fls,s3_sencing,s4_senced,s5_none]:
        model.add_state(s)
        model.add_transition(model.start, s, 1)
        
    model.add_transitions(s0_emerge,  [s0_emerge, s1_growth, s2_fls, s3_sencing, s4_senced, s5_none],
                                      [90.,       5.,        0.,     0.,         0.,        5.])
    model.add_transitions(s1_growth,  [s0_emerge, s1_growth, s2_fls, s3_sencing, s4_senced, s5_none],
                                      [0.,        90.,       2.5,    2.5,        0.,        5.])
    model.add_transitions(s2_fls,     [s0_emerge, s1_growth, s2_fls, s3_sencing, s4_senced, s5_none],
                                      [0.,        0.,        90.,    5.,         0.,        5.])
    model.add_transitions(s3_sencing, [s0_emerge, s1_growth, s2_fls, s3_sencing, s4_senced, s5_none],
                                      [0. ,       0.,        0.,     90.,        5.,        5.])
    model.add_transitions(s4_senced,  [s0_emerge, s1_growth, s2_fls, s3_sencing, s4_senced, s5_none],
                                      [0.,        0.,        0.,     0.,         90.,       10.])
    model.add_transitions(s5_none,    [s0_emerge, s1_growth, s2_fls, s3_sencing, s4_senced, s5_none],
                                      [10.,       0,        0.,     0.,         0.,        90.])
        
    model.bake(verbose=False)
    
    return model
