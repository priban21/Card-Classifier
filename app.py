import streamlit as st
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf

model = joblib.load("C:/Users/PBanerjee/Desktop/DIGI_PROJ/myenv/resnet50.joblib")

def preprocessing(img):
    processed_image = img.resize((255,255))
    processed_image = np.array(processed_image) / 255.0  # Normalize
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image



st.title("ID card classifier")
st.write("Upload an image of Aadhaar or Pan card")


file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])


if file is not None :
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify"):
        processed_image = preprocessing(image)
        prediction = model.predict(processed_image)[0][0]
        if prediction >= 0.5:
            st.success("This is PAN Card")
        else:
            st.success("This is Aadhaar Card")