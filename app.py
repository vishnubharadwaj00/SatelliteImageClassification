from fastai.vision import open_image, load_learner, image, torch
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
from PIL import Image
import requests
from io import BytesIO

st.title(':satellite: Satellite Image Classification')

st.write("A simple web application to perform multi-class classification of Satellite Images.")
st.write("(Powered by PyTorch,fastai and Streamlit)")
st.write("Some of the classes include:")
classes = ['Cloudy', 'Partly Cloudy', 'Hazy',
           'Primary (Rainforest)', 'Water', 'Habitation', 'Agriculture', 'Road', 'Cultivation']
st.write(classes)


@st.cache(suppress_st_warning=True)
def predict(img, display_img):

    # Display the test image
    st.image(display_img, use_column_width=True)

    # Temporarily displays a message while executing
    with st.spinner('Wait for it...'):
        time.sleep(3)

    # Load model and make prediction
    model = load_learner(path='', file='export.pkl')
    pred_class = model.predict(img)[0]
    output = str(pred_class)
    finalout = output.split(sep=';')

    st.write("The classes are:")
    st.write(finalout)


st.header("Pick an option:")
option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':

    # Test image selection
    test_images = os.listdir('images')
    test_image = st.selectbox(
        'Please select a test image: (If you are unable to see the image, pick another one)', test_images)

    # Read the image
    file_path = 'images/' + test_image
    img = open_image(file_path)
    # Get the image to display
    display_img = mpimg.imread(file_path)

    # Predict and display the image
    predict(img, display_img)

else:
    url = st.text_input("Please input a valid URL below:")

    st.write("(URL should lead to a satellite image of JPG or PNG format)")

    if url != "":
        try:
            # Read image from the url
            response = requests.get(url)
            pil_img = Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img)  # Image to display

            # Transform the image to feed into the model
            img = pil_img.convert('RGB')
            img = image.pil2tensor(img, np.float32).div_(255)
            img = image.Image(img)

            # Predict and display the image
            predict(img, display_img)

        except:
            st.write('Invalid URL')
