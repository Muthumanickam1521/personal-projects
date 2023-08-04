import streamlit as st
import numpy as np
from PIL import Image
from image_classifier import wolf_or_husky_classifier
from tensorflow.keras.utils import load_img, img_to_array

st.title('Wolf Vs. Husky Image Classification') # page title
st.image('background_image.jpeg')

image_path = st.file_uploader('Upload image') # widget to put media file

if image_path is not None:
    image_file = Image.open(image_path)
    st.image(image_file, caption='Image uploaded successfully') # displays image

    image_array = img_to_array(image_file) # converts pixels to numpy arrays
    image_array = np.expand_dims(image_array, axis = 0)

    pred = wolf_or_husky_classifier(image_array, 'model_weights.npy')
    
    if np.argmax(pred) == 0:
        label = 'Given image is husky!'
    else:
        label = 'Given image is wolf!'

    st.write(label)
 
else:
    st.write('Upload image again')

