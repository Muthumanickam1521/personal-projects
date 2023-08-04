import keras
import numpy as np
from tensorflow.keras import models, layers, Sequential

def wolf_or_husky_classifier(image, model_weights_path):
    '''	
    A simple image classifier function that takes 
    image path and model weights path as inputs
    '''
    weights = np.load(model_weights_path, allow_pickle = True) # loading pre_trained model weights
    
    resize_and_normalize = Sequential([ # model with two pre-processing layers
    	layers.Resizing(256, 256), # layer1: resize
    	layers.Rescaling(1/255) # layer2: rescale
    ])
    
    model = Sequential([ 
	  resize_and_normalize,
	  layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
	  layers.MaxPooling2D(),
	  layers.Flatten(),
	  layers.Dense(128, activation = 'relu')
	]) # classifier model architecture
    model.build(input_shape = (None, 256, 256, 3)) # build model 
    model.set_weights(weights) # put weights to model
    
    pred = model.predict(image) # prediction happens here
    
    return pred

