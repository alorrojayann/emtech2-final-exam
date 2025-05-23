import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# loading the model
@st.cache_resource
def load_smoker_model():
    model = load_model("smoker_model.h5", compile=False)
    return model

model = load_smoker_model()

class_names = ['Not Smoking', 'Smoking']  # adjust order if your training labels were reversed

st.title("Smoking Detection App")
st.write("Upload an image to detect if a person is smoking.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess the image
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### Prediction: `{predicted_class}`")
    st.markdown(f"### Confidence: `{confidence:.2f}`")
